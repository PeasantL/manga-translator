from dotenv import load_dotenv

load_dotenv()
import argparse
import ast
import asyncio
import os
from translator.pipeline import FullConversion, draw_page, align_translations
from translator.plugins import (
    get_translators,
    get_ocr,
    get_drawers,
    get_cleaners,
)
from translator.chapter import (
    find_chapters,
    build_document,
    Chapter,
    ChapterDocument,
)
from translator.plugins import OcrResult, TranslatorResult
from translator.utils import read_image, write_image


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def convert_to_options_list(classes: list):
    result = ""
    for x in range(len(classes)):
        item = classes[x]
        result += f"{x}) {item.__name__} => {item.__doc__}\n"

    return result[:-1]


def json_to_args(args_str: str):
    args = {}
    for item in args_str.strip().split(","):
        if "=" not in item:
            continue
        a = item.strip()
        equ_idx = a.index("=")
        key = a[0:equ_idx]
        raw = a[equ_idx + 1:].strip()
        try:
            # Lets 'size=12' and "text='hi'" both work, without eval's ability to
            # run arbitrary code from the command line.
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Anything that is not a Python literal is taken as a plain string,
            # which is what 'key=value' looks like to everyone but Python.
            value = raw
        args[key] = value
    return args


def select_plugin(options: list, index: int, kind: str):
    if index < 0 or index >= len(options):
        raise SystemExit(
            f"Invalid {kind} index {index}, must be between 0 and {len(options) - 1}"
        )
    return options[index]


STAGES = ["all", "ocr", "translate", "draw"]


def load_pages(chapter: Chapter) -> list[tuple[str, "object"]]:
    """Read a chapter's pages, dropping any the decoder cannot handle."""
    loaded = []

    for path in chapter.pages:
        frame = read_image(path)

        if frame is None:
            print(f"Skipping {path}, could not be read as an image")
        else:
            loaded.append((path, frame))

    return loaded


async def stage_ocr(chapter: Chapter, args):
    """Steps 1 to 4: detect, segment, clean, read.

    Writes the cleaned pages and ocr.json. No translator is built, so this stage
    needs no API key.
    """
    loaded = load_pages(chapter)

    if len(loaded) == 0:
        print(f"{chapter.name}: no readable pages")
        return

    converter = FullConversion(
        ocr=select_plugin(get_ocr(), args.ocr, "ocr")(**json_to_args(args.ocr_args)),
        cleaner=select_plugin(get_cleaners(), args.cleaner, "cleaner")(
            **json_to_args(args.cleaner_args)
        ),
    )

    print(f"{chapter.name}: reading {len(loaded)} pages")

    pages, ocr_results = await converter.clean_and_read(
        [frame for _, frame in loaded],
        names=[os.path.basename(path) for path, _ in loaded],
    )

    os.makedirs(chapter.clean_dir, exist_ok=True)

    # Always PNG here, whatever the source was. The cleaned page is an
    # intermediate that gets read back and drawn on, and beside the output
    # folder there is no reason for it to be anything but lossless -- unlike
    # the copy the service carries inside an archive it hands to a library,
    # where the size of it is somebody else's disk.
    document = build_document(
        chapter.name,
        pages,
        ocr_results,
        names=[os.path.basename(path) for path, _ in loaded],
        sources=[path.replace("\\", "/") for path, _ in loaded],
        clean_ext=".png",
    )

    for page, layout in zip(document.pages, pages):
        clean_path = os.path.join(chapter.output_dir, page.clean)

        if not write_image(clean_path, layout.frame):
            print(f"{chapter.name}: could not write {clean_path}")

    document.save(chapter.ocr_path)

    print(
        f"{chapter.name}: wrote {len(document.regions())} regions to "
        f"{chapter.ocr_path} and cleaned pages to {chapter.clean_dir}/"
    )


async def stage_translate(chapter: Chapter, args):
    """Step 5: translate ocr.json into translated.json.

    Reads and writes JSON only. No detection, cleaning or OCR model is loaded.
    """
    document = ChapterDocument.load(chapter.ocr_path)
    regions = document.regions()

    if len(regions) == 0:
        print(f"{chapter.name}: no regions to translate")
        document.save(chapter.translated_path)
        return

    translator = select_plugin(get_translators(), args.translator, "translator")(
        **json_to_args(args.translator_args)
    )

    print(f"{chapter.name}: translating {len(regions)} regions as one chapter")

    translations = align_translations(
        list(await translator([OcrResult(r.text, r.language) for r in regions])),
        len(regions),
    )

    for region, translation in zip(regions, translations):
        region.translation = translation.text

    document.target_language = getattr(translator, "target_lang", "")
    document.save(chapter.translated_path)

    print(f"{chapter.name}: wrote {chapter.translated_path}")


async def stage_draw(chapter: Chapter, args):
    """Step 6: draw translated.json onto the cleaned pages.

    Reads the cleaned pages and the JSON, so no vision or language model is
    loaded here either. The finished pages go in their own folder next to the
    cleaned ones, so that what this stage produced can be handed over, deleted or
    re-made without touching the input to it.
    """
    document = ChapterDocument.load(chapter.translated_path)

    drawer = select_plugin(get_drawers(), args.drawer, "drawer")(
        **json_to_args(args.drawer_args)
    )

    os.makedirs(chapter.drawn_dir, exist_ok=True)

    written = 0

    for page in document.pages:
        clean_path = os.path.join(chapter.output_dir, page.clean)
        frame = read_image(clean_path)

        if frame is None:
            print(f"{chapter.name}: missing cleaned page {clean_path}, skipping")
            continue

        drawn = await draw_page(
            frame,
            [tuple(r.box) for r in page.regions],
            [
                TranslatorResult(r.translation, document.target_language)
                for r in page.regions
            ],
            drawer,
            [(r.text_color, r.background_color) for r in page.regions],
            [r.outlined for r in page.regions],
        )

        if write_image(os.path.join(chapter.drawn_dir, page.name), drawn):
            written += 1
        else:
            print(f"{chapter.name}: could not write {page.name}")

    print(f"{chapter.name}: wrote {written} pages to {chapter.drawn_dir}/")


async def do_convert(chapters: list, args):
    """Run the requested stage over every chapter.

    "all" runs the three stages in sequence rather than keeping the chapter in
    memory, so that a full run leaves behind exactly the same artifacts a staged
    run does, and the two paths cannot drift apart.
    """
    stages = STAGES[1:] if args.stage == "all" else [args.stage]

    for chapter in chapters:
        os.makedirs(chapter.output_dir, exist_ok=True)

        for stage in stages:
            try:
                if stage == "ocr":
                    await stage_ocr(chapter, args)
                elif stage == "translate":
                    await stage_translate(chapter, args)
                else:
                    await stage_draw(chapter, args)
            except (FileNotFoundError, ValueError) as problem:
                # A missing artifact, or one written by a build that used a
                # different schema. Either way the later stages have nothing to
                # work from, so stop on this chapter rather than half convert it.
                print(f"{chapter.name}: {problem}")
                break


def main():
    parser = argparse.ArgumentParser(
        prog="Manga Translator",
        description="Translates Manga Pages",
        formatter_class=SmartFormatter,
        exit_on_error=True
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="Path to a folder of chapter folders, or a list of images",
    )

    parser.add_argument(
        "-s",
        "--stage",
        default="all",
        choices=STAGES,
        help="R|Which part of the pipeline to run\n"
             "all) every stage, in order\n"
             "ocr) steps 1-4, writes the cleaned pages and ocr.json\n"
             "translate) step 5, writes translated.json from ocr.json\n"
             "draw) step 6, draws translated.json onto the cleaned pages",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--ocr",
        default=0,
        type=int,
        help="R|Set the index of the ocr class to use. must be one of the following\n"
             + convert_to_options_list(get_ocr()),
        required=False,
    )

    parser.add_argument(
        "-oa",
        "--ocr-args",
        default="",
        type=str,
        help="Set ocr class args i.e. 'key=value , key2=value'",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--translator",
        default=0,
        type=int,
        help="R|Set the index of the translator class to use. must be one of the following\n"
             + convert_to_options_list(get_translators()),
        required=False,
    )

    parser.add_argument(
        "-ta",
        "--translator-args",
        default="",
        type=str,
        help="Set translator class args i.e. 'key=value , key2=value'",
        required=False,
    )

    parser.add_argument(
        "-dr",
        "--drawer",
        default=0,
        type=int,
        help="R|Set the index of the drawer class to use. must be one of the following\n"
             + convert_to_options_list(get_drawers()),
        required=False,
    )

    parser.add_argument(
        "-dra",
        "--drawer-args",
        default="",
        type=str,
        help="Set drawer class args i.e. 'key=value , key2=value'",
        required=False,
    )

    parser.add_argument(
        "-c",
        "--cleaner",
        default=0,
        type=int,
        help="R|Set the index of the cleaner class to use. must be one of the following\n"
             + convert_to_options_list(get_cleaners()),
        required=False,
    )

    parser.add_argument(
        "-ca",
        "--cleaner-args",
        default="",
        type=str,
        help="Set cleaner class args i.e. 'key=value , key2=value'",
        required=False,
    )

    args = parser.parse_args()

    if args.files is None:
        parser.print_help()
        return

    chapters = find_chapters(args.files)

    if len(chapters) == 0:
        print("No chapters to convert")
        return

    print(f"Found {len(chapters)} chapter(s): {', '.join(c.name for c in chapters)}")

    asyncio.run(do_convert(chapters, args))


if __name__ == "__main__":
    main()
