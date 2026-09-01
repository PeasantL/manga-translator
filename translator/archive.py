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

import json
import posixpath
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from translator.chapter import CLEAN_DIR, IMAGE_EXTENSIONS, ChapterDocument
from translator.utils import natural_sort_key

COMICINFO = "ComicInfo.xml"

# What a translated archive is tagged with, and what marks one as already done
# so a library can tell the two apart without opening every file.
TRANSLATED_TAG = "translated"

# And the same fact as a genre, which is not a duplicate of it.
#
# A library reading ComicInfo files the two fields in different places: Tags
# describe the book and Genre describes the work, so in Komga the tag lands on
# the book record and the genre on the series wrapping it. A shelf of covers is
# series -- the books are a click further in -- so the tag alone is invisible
# at exactly the place a "this one is the translation" mark is worth having.
#
# Title case because a genre is displayed as written, alongside the ones the
# book came with.
TRANSLATED_GENRE = "Translated"

# Where a translated archive carries what it took to make it: the pages with
# the original text erased, and the document saying what was said in each
# bubble and what it became. Together they are everything stage 6 needs, so
# the lettering can be corrected and redrawn without the chapter going near a
# model again -- and without needing the untranslated book still to be there.
#
# Under a folder named for this project, so it is obvious what wrote them.
SIDECAR = "translator"
DOCUMENT = f"{SIDECAR}/chapter.json"

# The cleaned pages go in an archive of their own inside the outer one, and
# that nesting is the whole reason this is safe to do. A comic reader decides
# what the pages of a CBZ are by walking its entries and taking the images, at
# whatever depth it finds them -- so cleaned pages sitting loose in here would
# be counted as pages of the book, and every chapter would appear to have twice
# as many as it does. Nothing descends into a nested zip.
CLEAN_ARCHIVE = f"{SIDECAR}/clean.zip"


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

    # This project's own sidecar. Nothing in there is a page of the book, and
    # the nesting above already keeps the cleaned pages out of reach -- but a
    # translated archive gets unpacked again if someone translates it twice,
    # and saying so here is what makes that safe however the sidecar is laid
    # out later.
    if name.startswith(f"{SIDECAR}/"):
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
    title: str = "",
) -> bytes:
    """The source's ComicInfo, rewritten to describe the translation.

    Everything the original said is kept -- artist, circle, tags, the gallery it
    came from -- because the translation is the same book. Only what is no
    longer true changes: the language, and the title it is filed under.

    A `title` replaces the original outright -- the dialogue is in English now,
    so a Japanese title on the shelf beside it helps nobody. Without one the
    original is kept as it was, which is what happens when there was nothing to
    translate or translating it failed.

    Nothing is appended to say this is a translation. LanguageISO already says
    so, and a reader that shows the language has no use for the same fact
    spelled out again in the title.

    That it is one is said twice elsewhere, in Tags and in Genre, and those are
    not a duplicate of each other: a library reading this files the two in
    different places, so which of them can be seen depends on where you are
    looking from. See TRANSLATED_GENRE.

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

    if title:
        for tag in ("Title", "Series"):
            _child(root, tag).text = title

    _child(root, "LanguageISO").text = target_lang

    tags = _csv_field(root, "Tags")

    if TRANSLATED_TAG not in [t.lower() for t in tags]:
        tags.append(TRANSLATED_TAG)

    _child(root, "Tags").text = ", ".join(tags)

    genres = _csv_field(root, "Genre")

    if TRANSLATED_GENRE.lower() not in [g.lower() for g in genres]:
        genres.append(TRANSLATED_GENRE)

    _child(root, "Genre").text = ", ".join(genres)

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


def pack(
    pages: Path,
    dest: Path,
    comicinfo: bytes | None = None,
    extras: dict[str, bytes | Path] | None = None,
) -> int:
    """Zip a folder of finished pages into a CBZ. Returns the page count.

    `extras` are entries carried alongside the pages, given either as bytes or
    as a file to copy in -- the sidecar, in practice. They are written under
    the names they are keyed by, so it is the caller that decides a cleaned
    page cannot be mistaken for a page of the book.
    """
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

        for name, content in (extras or {}).items():
            if isinstance(content, (bytes, bytearray)):
                archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
            else:
                # A file rather than bytes is the inner archive, which is
                # already as small as it is going to get.
                archive.write(content, name, compress_type=zipfile.ZIP_STORED)

    return len(found)


def pack_clean(pages: Path, dest: Path) -> int:
    """Zip a folder of cleaned pages into the inner archive a CBZ carries.

    Entries keep the `clean/` prefix the document refers to them by, so that one
    document describes both the folder the CLI leaves behind and the archive the
    service embeds. Returns the page count.
    """
    found = sorted(
        (p for p in pages.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: natural_sort_key(p.name),
    )

    if len(found) == 0:
        raise ValueError(f"{pages} has no cleaned pages to pack")

    with zipfile.ZipFile(dest, "w") as archive:
        for page in found:
            archive.write(
                page, f"{CLEAN_DIR}/{page.name}", compress_type=zipfile.ZIP_STORED
            )

    return len(found)


def read_document(cbz: Path) -> ChapterDocument | None:
    """The chapter document a translated archive carries, if it has one.

    None means there is nothing in there to read -- an archive from before this
    was written, or one that never came from here. A document that is present
    but unreadable raises instead: that is a different problem and deserves to
    be said out loud rather than to look like an old file.
    """
    try:
        with zipfile.ZipFile(cbz) as archive:
            raw = archive.read(DOCUMENT)
    except KeyError:
        return None
    except (zipfile.BadZipFile, OSError) as problem:
        raise ValueError(f"{cbz.name} could not be opened: {problem}") from problem

    try:
        return ChapterDocument.from_dict(json.loads(raw))
    except json.JSONDecodeError as problem:
        raise ValueError(f"{cbz.name} has a chapter document that will not parse") from problem


def extract_clean_archive(cbz: Path, dest: Path) -> Path | None:
    """Copy the inner archive of cleaned pages out of a CBZ. None if it has none.

    Copied to a file rather than read into memory: it holds every page of the
    chapter, and a zip has to be seekable to be opened at all, which a stream
    out of the archive around it is not.
    """
    with zipfile.ZipFile(cbz) as archive:
        try:
            entry = archive.open(CLEAN_ARCHIVE)
        except KeyError:
            return None

        with entry, dest.open("wb") as out:
            shutil.copyfileobj(entry, out)

    return dest


def unpack_clean(clean_zip: Path, dest: Path) -> dict[str, Path]:
    """Extract cleaned pages into dest, keyed by the name a document finds them by.

    Flattened onto the basename of each entry, for the reason the module
    docstring gives: an entry name is written by whoever packed the file, and
    `../../etc/passwd` is a valid one. A document's `clean` field is read the
    same way, so the two still meet.
    """
    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}

    with zipfile.ZipFile(clean_zip) as archive:
        for name in archive.namelist():
            base = posixpath.basename(name)

            if not base or base.startswith("."):
                continue

            page = dest / base

            with archive.open(name) as source, page.open("wb") as out:
                shutil.copyfileobj(source, out)

            found[base] = page

    return found
