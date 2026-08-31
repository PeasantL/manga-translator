"""Chapters: a folder of pages in, a folder of the same name out.

A chapter is a folder of images under the input root. Loose images sitting
directly in the input root are not a chapter and are skipped - a chapter needs a
folder so that its results have a folder of the same name to go into:

    input/                        output/
        my-oneshot/                   my-oneshot/
            01.png                        clean/01.png
            02.png                        clean/02.png
                                          ocr.json
                                          translated.json
                                          01.png
                                          02.png
"""

import os

from translator.utils import natural_sort_key

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

CLEAN_DIR = "clean"
OCR_FILE = "ocr.json"
TRANSLATED_FILE = "translated.json"


class Chapter:
    """One folder of pages, and where its results go."""

    def __init__(self, name: str, pages: list[str], output_dir: str) -> None:
        self.name = name
        self.pages = pages
        self.output_dir = output_dir

    @property
    def clean_dir(self) -> str:
        return os.path.join(self.output_dir, CLEAN_DIR)

    @property
    def ocr_path(self) -> str:
        return os.path.join(self.output_dir, OCR_FILE)

    @property
    def translated_path(self) -> str:
        return os.path.join(self.output_dir, TRANSLATED_FILE)

    def __repr__(self) -> str:
        return f"Chapter({self.name!r}, {len(self.pages)} pages)"


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def pages_in(folder: str) -> list[str]:
    """The images directly inside a folder, in reading order.

    Sorted naturally so that page2 comes before page10. Subfolders are not
    descended into: one folder is one chapter.
    """
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder), key=natural_sort_key)
        if is_image(os.path.join(folder, name))
    ]


def find_chapters(paths: list[str], output_root: str = "output") -> list[Chapter]:
    """Resolve the -f argument into chapters.

    A single folder is an input root: each of its subfolders is a chapter, and
    any images lying loose in the root are skipped. An explicit list of files is
    grouped into chapters by the folder each file sits in.
    """
    chapters = []

    if len(paths) == 1 and os.path.isdir(paths[0]):
        root = paths[0]
        loose = []

        for name in sorted(os.listdir(root), key=natural_sort_key):
            entry = os.path.join(root, name)

            if os.path.isdir(entry):
                pages = pages_in(entry)

                if len(pages) == 0:
                    print(f"Skipping {entry}, no images in it")
                else:
                    chapters.append(
                        Chapter(name, pages, os.path.join(output_root, name))
                    )
            elif is_image(entry):
                loose.append(name)

        if len(loose) > 0:
            print(
                f"Skipping {len(loose)} loose image(s) in {root}: "
                "put each chapter in its own folder"
            )
    else:
        # An explicit list of files. Group by the folder each one is in so that
        # the chapter still has a name to give its output folder.
        grouped: dict[str, list[str]] = {}

        for path in paths:
            if not os.path.isfile(path):
                print(f"Skipping {path}, not a file")
            elif not is_image(path):
                print(f"Skipping {path}, not a supported image type")
            else:
                name = os.path.basename(os.path.dirname(os.path.abspath(path)))
                grouped.setdefault(name, []).append(path)

        for name, pages in grouped.items():
            chapters.append(Chapter(name, pages, os.path.join(output_root, name)))

    return chapters
