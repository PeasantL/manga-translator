from dotenv import load_dotenv

load_dotenv()
import argparse
import ast
import cv2
import asyncio
import os
from translator.pipelines import FullConversion
from translator.translators.get import get_translators
from translator.ocr.get import get_ocr
from translator.drawers.get import get_drawers
from translator.cleaners.get import get_cleaners
from translator.chapter import find_chapters


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


async def do_convert(chapters: list, translator: int, translator_args: str, ocr: int, ocr_args: str, drawer: int, drawer_args: str, cleaner: int, cleaner_args: str):
    converter = FullConversion(
        translator=select_plugin(get_translators(), translator, "translator")(**json_to_args(translator_args)),
        ocr=select_plugin(get_ocr(), ocr, "ocr")(**json_to_args(ocr_args)),
        drawer=select_plugin(get_drawers(), drawer, "drawer")(**json_to_args(drawer_args)),
        cleaner=select_plugin(get_cleaners(), cleaner, "cleaner")(**json_to_args(cleaner_args)),
    )

    for chapter in chapters:
        await convert_chapter(chapter, converter)


def load_pages(chapter) -> list[tuple[str, "object"]]:
    """Read a chapter's pages, dropping any the decoder cannot handle."""
    loaded = []

    for path in chapter.pages:
        frame = cv2.imread(path)

        if frame is None:
            print(f"Skipping {path}, could not be read as an image")
        else:
            loaded.append((path, frame))

    return loaded


async def convert_chapter(chapter, converter: FullConversion):
    loaded = load_pages(chapter)

    if len(loaded) == 0:
        print(f"{chapter.name}: nothing to convert")
        return

    os.makedirs(chapter.output_dir, exist_ok=True)

    # The whole chapter is handed over at once, so the translator is given its
    # dialogue as a single ordered list and can use the surrounding lines for
    # context. FullConversion still chunks the models it runs page by page.
    print(f"{chapter.name}: converting {len(loaded)} pages as one chapter")

    results = await converter([frame for _, frame in loaded])

    for (path, _), frame in zip(loaded, results):
        cv2.imwrite(os.path.join(chapter.output_dir, os.path.basename(path)), frame)

    print(f"{chapter.name}: wrote {len(loaded)} pages to {chapter.output_dir}/")


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

    asyncio.run(do_convert(
        chapters,
        args.translator,
        args.translator_args,
        args.ocr,
        args.ocr_args,
        args.drawer,
        args.drawer_args,
        args.cleaner,
        args.cleaner_args,
    ))


if __name__ == "__main__":
    main()
