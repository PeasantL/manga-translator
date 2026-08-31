from dotenv import load_dotenv

load_dotenv()
import argparse
import ast
import cv2
import asyncio
import os
import math
from translator.pipelines import FullConversion
from translator.translators.get import get_translators
from translator.ocr.get import get_ocr
from translator.drawers.get import get_drawers
from translator.cleaners.get import get_cleaners

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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


def collect_image_files(paths: list[str]) -> list[str]:
    """Expand a folder argument into the image files inside it.

    A folder holding a README or a subdirectory used to take the whole run down,
    because the non-image entries reached cv2.imread as None.
    """
    if len(paths) == 1 and os.path.isdir(paths[0]):
        folder = paths[0]
        candidates = [os.path.join(folder, x) for x in sorted(os.listdir(folder))]
    else:
        candidates = paths

    files = []
    for path in candidates:
        if not os.path.isfile(path):
            print(f"Skipping {path}, not a file")
        elif os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            print(f"Skipping {path}, not a supported image type")
        else:
            files.append(path)

    return files


def select_plugin(options: list, index: int, kind: str):
    if index < 0 or index >= len(options):
        raise SystemExit(
            f"Invalid {kind} index {index}, must be between 0 and {len(options) - 1}"
        )
    return options[index]


async def do_convert(files: list[str], translator: int, translator_args: str, ocr: int, ocr_args: str, drawer: int, drawer_args: str, cleaner: int, cleaner_args: str):
    converter = FullConversion(
        translator=select_plugin(get_translators(), translator, "translator")(**json_to_args(translator_args)),
        ocr=select_plugin(get_ocr(), ocr, "ocr")(**json_to_args(ocr_args)),
        drawer=select_plugin(get_drawers(), drawer, "drawer")(**json_to_args(drawer_args)),
        cleaner=select_plugin(get_cleaners(), cleaner, "cleaner")(**json_to_args(cleaner_args)),
    )

    filenames = files
    batches = math.ceil(len(filenames) / 4)
    output_dir = 'output'  # Define the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    for i in range(batches):
        files_to_convert = filenames[i * 4: (i + 1) * 4]

        loaded = []
        for file in files_to_convert:
            frame = cv2.imread(file)
            if frame is None:
                print(f"Skipping {file}, could not be read as an image")
            else:
                loaded.append((file, frame))

        if len(loaded) == 0:
            continue

        for filename, data in zip(
                [x[0] for x in loaded], await converter([x[1] for x in loaded])
        ):
            frame = data
            base, ext = os.path.splitext(os.path.basename(filename))
            new_filename = os.path.join(output_dir, base + "_converted" + (ext if ext else ".png"))
            cv2.imwrite(new_filename, frame)
        print(f"Converted Batch {i + 1}/{batches}")


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
        help="A list of images to convert or path to a folder of images",
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

    files = collect_image_files(args.files)

    if len(files) == 0:
        print("No images to convert")
        return

    asyncio.run(do_convert(
        files,
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
