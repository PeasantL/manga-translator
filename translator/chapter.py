"""Chapters: a folder of pages in, a folder of the same name out.

A chapter is a folder of images under the input root. Loose images sitting
directly in the input root are not a chapter and are skipped - a chapter needs a
folder so that its results have a folder of the same name to go into:

    input/                        output/
        my-oneshot/                   my-oneshot/
            01.png                        clean/01.png
            02.png                        clean/02.png
                                          drawn/01.png
                                          drawn/02.png
                                          ocr.json
                                          translated.json
"""

import json
import os

from translator.utils import natural_sort_key

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

CLEAN_DIR = "clean"
DRAWN_DIR = "drawn"
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
    def drawn_dir(self) -> str:
        return os.path.join(self.output_dir, DRAWN_DIR)

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


class Region:
    """One bubble: where it is, what it said, and what that becomes.

    `box` is the area the translation is drawn into, in the page's pixel
    coordinates, which is all stage 6 needs to place text without redetecting.
    The colours are measured while the original text is still on the page, so
    that stage 6 can letter a white on black bubble without looking at the art.
    """

    def __init__(
        self,
        box: list[int],
        text: str = "",
        language: str = "",
        translation: str = "",
        text_color: list[int] = None,
        background_color: list[int] = None,
    ) -> None:
        self.box = [int(v) for v in box]
        self.text = text
        self.language = language
        self.translation = translation
        # Both BGR, like everything else that came out of OpenCV, and both
        # measured off the page while the text was still there. null means the
        # cleaner found no glyphs to measure.
        self.text_color = list(text_color) if text_color is not None else None
        self.background_color = (
            list(background_color) if background_color is not None else None
        )

    def to_dict(self) -> dict:
        return {
            "box": self.box,
            "text": self.text,
            "language": self.language,
            "translation": self.translation,
            "text_color": self.text_color,
            "background_color": self.background_color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Region":
        return cls(
            box=data["box"],
            text=data.get("text", ""),
            language=data.get("language", ""),
            translation=data.get("translation", ""),
            text_color=data.get("text_color"),
            background_color=data.get("background_color"),
        )


class Page:
    """One page of a chapter. `clean` is relative to the chapter's output folder
    so the folder can be moved or handed to someone else whole."""

    def __init__(
        self,
        name: str,
        source: str,
        clean: str,
        width: int,
        height: int,
        regions: list[Region],
    ) -> None:
        self.name = name
        self.source = source
        self.clean = clean
        self.width = width
        self.height = height
        self.regions = regions

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "clean": self.clean,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        return cls(
            name=data["name"],
            source=data.get("source", ""),
            clean=data["clean"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            regions=[Region.from_dict(r) for r in data.get("regions", [])],
        )


class ChapterDocument:
    """What one stage hands the next.

    Pages are in reading order and so are the regions within each page, so
    flattening them gives the chapter's dialogue in sequence - which is the order
    the translator needs to make sense of it.
    """

    SCHEMA = 2

    def __init__(
        self,
        chapter: str,
        source_language: str = "",
        target_language: str = "",
        pages: list[Page] = None,
    ) -> None:
        self.chapter = chapter
        self.source_language = source_language
        self.target_language = target_language
        self.pages = pages if pages is not None else []

    def regions(self) -> list[Region]:
        return [region for page in self.pages for region in page.regions]

    def to_dict(self) -> dict:
        return {
            "schema": ChapterDocument.SCHEMA,
            "chapter": self.chapter,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChapterDocument":
        schema = data.get("schema")

        if schema != ChapterDocument.SCHEMA:
            raise ValueError(
                f"Unsupported schema {schema}, this build writes and reads "
                f"schema {ChapterDocument.SCHEMA}"
            )

        return cls(
            chapter=data.get("chapter", ""),
            source_language=data.get("source_language", ""),
            target_language=data.get("target_language", ""),
            pages=[Page.from_dict(p) for p in data.get("pages", [])],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ChapterDocument":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} is missing. Run the earlier stage first."
            )

        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
