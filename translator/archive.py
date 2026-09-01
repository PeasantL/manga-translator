"""CBZ in, CBZ out: the archive end of a translation job.

The pipeline works on a chapter folder (translator/chapter.py); a library holds
CBZs. This is the conversion between the two, and the only place here that knows
an archive is a zip.

Pages are extracted under sequential names rather than the ones inside the
archive, for two reasons. An archive's own names sort lexically wherever
something other than this project reads them, so `10.webp` lands before
`2.webp` and the chapter is read out of order. And an entry name is written by
whoever packed the file -- `../../etc/passwd` is a valid zip entry. Never
letting those names reach the filesystem removes both problems at once.
"""

import posixpath
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from translator.chapter import IMAGE_EXTENSIONS
from translator.utils import natural_sort_key

COMICINFO = "ComicInfo.xml"

# What a translated archive is tagged with, and what marks one as already done
# so a library can tell the two apart without opening every file.
TRANSLATED_TAG = "translated"


def _is_page(name: str) -> bool:
    """Whether a zip entry is one of the chapter's pages."""
    if name.endswith("/"):
        return False

    base = posixpath.basename(name)

    # __MACOSX holds resource forks that pair with every real entry and decode
    # as images; a leading dot is the other thing packers leave behind. Neither
    # is a page, and both would be drawn into the output if counted as one.
    if not base or base.startswith(".") or name.startswith("__MACOSX/"):
        return False

    return posixpath.splitext(base)[1].lower() in IMAGE_EXTENSIONS


def page_entries(archive: zipfile.ZipFile) -> list[str]:
    """The archive's pages, in reading order.

    Sorted naturally over the whole path rather than the basename, so that an
    archive which nests its pages in a folder keeps the folders in order and
    still gets page2 before page10 inside each one.
    """
    return sorted((n for n in archive.namelist() if _is_page(n)), key=natural_sort_key)


def unpack(cbz: Path, dest: Path) -> list[Path]:
    """Extract a CBZ's pages into dest as a chapter folder, in reading order."""
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cbz) as archive:
        names = page_entries(archive)

        if len(names) == 0:
            raise ValueError(f"{cbz.name} has no pages in it")

        # Zero padded to the width the count needs so that the names the
        # pipeline goes on to write sort correctly in a reader that sorts
        # lexically, which is most of them.
        width = len(str(len(names)))
        written = []

        for index, name in enumerate(names, start=1):
            page = dest / f"{index:0{width}d}{posixpath.splitext(name)[1].lower()}"

            # Copied in chunks rather than read whole: a page is a few megabytes
            # and a long chapter is hundreds of them.
            with archive.open(name) as source, page.open("wb") as out:
                shutil.copyfileobj(source, out)

            written.append(page)

    return written


def read_comicinfo(cbz: Path) -> bytes | None:
    """The archive's ComicInfo.xml, if it has one.

    Matched at any depth, because an archive that nests its pages in a folder
    usually nests this alongside them. Returns None rather than raising for
    anything unexpected: a chapter with no metadata still translates fine, it
    just has nothing to carry over.
    """
    try:
        with zipfile.ZipFile(cbz) as archive:
            name = next(
                (
                    n
                    for n in archive.namelist()
                    if posixpath.basename(n).lower() == COMICINFO.lower()
                ),
                None,
            )

            return None if name is None else archive.read(name)
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


def _child(root: ET.Element, tag: str) -> ET.Element:
    """The named child, created empty if it is not there yet."""
    found = root.find(tag)

    return ET.SubElement(root, tag) if found is None else found


def _csv_field(root: ET.Element, tag: str) -> list[str]:
    """A comma separated ComicInfo field as a list."""
    node = root.find(tag)

    if node is None or not node.text:
        return []

    return [part.strip() for part in node.text.split(",") if part.strip()]


def source_title(original: bytes | None) -> str:
    """The Title the archive's ComicInfo carries, if it has one."""
    if not original:
        return ""

    try:
        root = ET.fromstring(original)
    except ET.ParseError:
        return ""

    node = root.find("Title")

    return (node.text or "").strip() if node is not None else ""


def translated_comicinfo(
    original: bytes | None,
    target_lang: str = "en",
    title_suffix: str = "[EN]",
    title: str = "",
) -> bytes:
    """The source's ComicInfo, rewritten to describe the translation.

    Everything the original said is kept -- artist, circle, tags, the gallery it
    came from -- because the translation is the same book. Only what is no
    longer true changes: the language, and the title it is filed under.

    A `title` replaces the original outright -- the dialogue is in English now,
    so a Japanese title on the shelf beside it helps nobody. Without one the
    original is kept and only marked, which is what happens when there was
    nothing to translate or translating it failed.

    Built through ElementTree rather than string edits so that a title with an
    ampersand in it cannot produce a file Komga refuses to parse.
    """
    root = None

    if original:
        try:
            root = ET.fromstring(original)
        except ET.ParseError:
            # A ComicInfo that will not parse is not worth failing a finished
            # translation over. It gets a fresh one instead.
            root = None

    if root is None or root.tag != "ComicInfo":
        root = ET.Element("ComicInfo")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

    for tag in ("Title", "Series"):
        node = _child(root, tag) if title else root.find(tag)

        if node is None:
            continue

        if title:
            node.text = title

        # Guarded on the suffix already being there so that translating a file
        # twice does not produce "Name [EN] [EN]".
        if node.text and title_suffix not in node.text:
            node.text = f"{node.text} {title_suffix}"

    _child(root, "LanguageISO").text = target_lang

    tags = _csv_field(root, "Tags")

    if TRANSLATED_TAG not in [t.lower() for t in tags]:
        tags.append(TRANSLATED_TAG)

    _child(root, "Tags").text = ", ".join(tags)

    # Appended rather than replaced: Notes is where the download recorded which
    # gallery this came from, and that is still true of the translation.
    # Guarded on the marker rather than the whole sentence, which carries a date
    # and so would not match itself on a later day -- translating twice would
    # otherwise leave two of these behind, as the title would without its own
    # guard above.
    marker = f"Translated to {target_lang} by manga-translator"
    notes = _child(root, "Notes")

    if marker not in (notes.text or ""):
        note = f"{marker} on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        notes.text = f"{notes.text}. {note}" if notes.text else note

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def pack(pages: Path, dest: Path, comicinfo: bytes | None = None) -> int:
    """Zip a folder of finished pages into a CBZ. Returns the page count."""
    found = sorted(
        (p for p in pages.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: natural_sort_key(p.name),
    )

    if len(found) == 0:
        raise ValueError(f"{pages} has no pages to pack")

    with zipfile.ZipFile(dest, "w") as archive:
        for page in found:
            # Stored rather than deflated: PNG, JPEG and WebP are already
            # compressed, so deflating them again costs CPU over a whole
            # chapter and saves close to nothing. Flat rather than nested,
            # because a folder inside a CBZ means nothing to a reader.
            archive.write(page, page.name, compress_type=zipfile.ZIP_STORED)

        if comicinfo:
            # The one entry here that is text, and the one worth compressing.
            archive.writestr(COMICINFO, comicinfo, compress_type=zipfile.ZIP_DEFLATED)

    return len(found)
