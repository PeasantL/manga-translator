"""A job API over the pipeline: one CBZ in, one translated CBZ out.

This is how everything but the CLI reaches the pipeline: a library hands over a
whole chapter and collects a translated one back minutes later. That is far too
long to hold a request open for, so submitting a chapter starts a job and
returns immediately.

This process is the one that holds the job. Whoever submitted a chapter can go
away, restart, and come back to find the finished file still waiting, which is
what lets the caller keep no state of its own. A finished job is kept until it
is collected and deleted.

One job runs at a time. There is one GPU, the models are process-global, and a
second chapter running concurrently would only make both slower.
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from translator import archive
from translator.chapter import ChapterDocument, build_document
from translator.pipeline import (
    DETECT_REPO,
    SEGMENT_REPO,
    SEGMENT_WEIGHTS,
    FullConversion,
    draw_page,
)
from translator.plugins import (
    DebugTranslator,
    DeepSeekTranslator,
    HorizontalDrawer,
    ComicOcr,
    OcrResult,
    TranslatorResult,
)
from translator.plugins.cleaning import REPO as LAMA_REPO, WEIGHTS as LAMA_WEIGHTS
from translator.plugins.ocr import MODEL as OCR_REPO
from translator.utils import read_image, write_image

log = logging.getLogger("translator.service")

# The four sets of weights the pipeline runs on, each as the repository it
# comes from and one file that has to be in it. Enough to tell a cold cache
# from a warm one without loading anything.
WEIGHTS = [
    (DETECT_REPO, "model.safetensors"),
    (SEGMENT_REPO, SEGMENT_WEIGHTS),
    (LAMA_REPO, LAMA_WEIGHTS),
    (OCR_REPO, "model.safetensors"),
]

JOB_DIR = Path(os.environ.get("JOB_DIR", "jobs"))
TARGET_LANG = os.environ.get("TARGET_LANG", "en")

# What to assume a chapter is in when the caller does not say. A caller that
# knows -- a library holding the book's own LanguageISO -- sends it per job and
# overrides this; the prompt is picked from whichever it ends up being.
#
# Note that the OCR is manga-ocr and reads Japanese only, so a non-Japanese
# source currently gets the right prompt applied to text the reader could not
# make out. The prompt and the plumbing are here ahead of an OCR that can.
SOURCE_LANG = os.environ.get("SOURCE_LANG", "ja")

# "debug" writes a fixed string into every bubble and needs no API key, which
# is how detection, cleaning and lettering get exercised on their own -- worth
# having as a setting because it is also how this service is smoke tested.
TRANSLATORS = {"deepseek": DeepSeekTranslator, "debug": DebugTranslator}
TRANSLATOR = os.environ.get("TRANSLATOR", "deepseek").lower()

# Read in chunks rather than with .read(): a chapter is tens of megabytes and
# there is no reason for all of it to be resident at once.
UPLOAD_CHUNK = 1024 * 1024

# What the cleaned pages carried inside a result are written as.
#
# WebP rather than the lossless PNG the CLI leaves beside its output, because
# these are not an intermediate on this machine: they go into the archive that
# goes into somebody's library, and lossless would roughly double what a
# chapter costs to keep. Nothing accumulates from it either -- a redraw always
# starts from this same cleaned page, so however many times the lettering is
# corrected, it is encoded once.
#
# Set CLEAN_QUALITY above 100 for lossless, if the disk is cheaper than the
# doubt.
CLEAN_EXT = ".webp"
CLEAN_QUALITY = int(os.environ.get("CLEAN_QUALITY", "95"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any job directory still here belongs to a previous process, whose record
    # of it died with it. Nothing can collect those, so they are just disk.
    if JOB_DIR.exists():
        for stale in JOB_DIR.iterdir():
            shutil.rmtree(stale, ignore_errors=True)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="mangatranslate", lifespan=lifespan)

# The pipeline, built on first use and kept for the life of the process.
# Building it loads three models and costs the better part of ten seconds, so
# it is not worth doing per job -- but it is worth deferring until a job
# actually arrives, because until then it would hold GPU memory that anything
# else sharing the card could be using.
_pipeline: FullConversion | None = None

# The one job. None means nothing has been submitted since this process
# started; see the module docstring for why there is only ever one.
_job: dict | None = None


def get_pipeline() -> FullConversion:
    global _pipeline

    if _pipeline is None:
        if TRANSLATOR not in TRANSLATORS:
            raise ValueError(
                f"TRANSLATOR is {TRANSLATOR!r}, expected one of {sorted(TRANSLATORS)}"
            )

        log.info("building the pipeline, this loads the models")
        chosen = TRANSLATORS[TRANSLATOR]
        # DebugTranslator takes the text to write rather than a language, so
        # only the real one is told what to translate into.
        translator = (
            chosen(text="TRANSLATED") if chosen is DebugTranslator
            else chosen(target_lang=TARGET_LANG)
        )
        _pipeline = FullConversion(ocr=ComicOcr(), translator=translator)

    return _pipeline


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _public(job: dict) -> dict:
    """The job as the caller sees it, without this process's own bookkeeping."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/healthz")
async def healthz():
    """Whether this can take a job, and what it would run it on.

    Reports the models as present or not rather than loading them, so that a
    health check never has the side effect of pulling four gigabytes into
    memory, or of downloading them.
    """
    import torch
    from huggingface_hub import try_to_load_from_cache

    cached = [try_to_load_from_cache(repo, name) for repo, name in WEIGHTS]

    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": all(isinstance(path, str) for path in cached),
        "loaded": _pipeline is not None,
        "busy": _job is not None and _job["state"] == "running",
        "target_lang": TARGET_LANG,
        "source_lang": SOURCE_LANG,
        # Which sources the OCR can actually read, as opposed to which ones a
        # prompt exists for. A caller worth the name checks this before
        # sending something the reader will only turn into noise.
        "reads": list(ComicOcr.READS),
    }


@app.get("/jobs/current")
async def current_job():
    """What this is doing, if anything. The endpoint a caller polls."""
    return _public(_job) if _job else {"state": "idle"}


@app.get("/jobs/{job_id}")
async def one_job(job_id: str):
    if not _job or _job["id"] != job_id:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")

    return _public(_job)


def _claim() -> tuple[str, Path]:
    """Take the one job slot and hand back an id and a folder to work in.

    A finished job that nobody collected is replaced rather than protected: the
    caller polls, and one that has stopped polling long enough to submit
    something else is not coming back for it.
    """
    global _job

    if _job and _job["state"] == "running":
        raise HTTPException(status_code=409, detail="a job is already running")

    if _job:
        shutil.rmtree(_job["_dir"], ignore_errors=True)
        # Cleared rather than left pointing at a folder that is no longer
        # there, because the upload that follows is awaited and anything
        # polling during it would otherwise be told about a result it could
        # not collect.
        _job = None

    job_id = uuid.uuid4().hex
    workdir = JOB_DIR / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    return job_id, workdir


async def _receive(file: UploadFile, dest: Path) -> None:
    """Stream an upload to disk, taking its work folder down with it if it fails."""
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                out.write(chunk)
    except Exception:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise


def _record(job_id: str, workdir: Path, kind: str, name: str, filename: str, **extra) -> dict:
    """The job as it starts out. What both kinds have in common, plus theirs."""
    return {
        "id": job_id,
        "kind": kind,
        "name": name,
        "state": "running",
        "stage": "unpack",
        "done": 0,
        "total": 0,
        "detail": "opening the archive",
        "pages": 0,
        "regions": 0,
        "started": _now(),
        "finished": None,
        "_dir": workdir,
        "_result": None,
        # What the finished file is called when it is collected. Decided here
        # because only the endpoint that started the job knows: a translation
        # is a new book and gets a new name, a redraw replaces one and keeps
        # the name it arrived with.
        "_filename": filename,
        **extra,
    }


@app.post("/jobs")
async def submit(
    file: UploadFile = File(...), name: str = Form(""), source_lang: str = Form("")
):
    """Take a CBZ and start translating it."""
    global _job

    job_id, workdir = _claim()

    source = (source_lang or SOURCE_LANG).strip().lower()

    if source not in ComicOcr.READS:
        # Not refused: the prompt for it exists and the rest of the pipeline
        # runs, so this is a warning about the quality of the result rather
        # than a reason not to produce one.
        log.warning(
            "job asked for %s, which the OCR cannot read -- it reads %s",
            source,
            ", ".join(sorted(ComicOcr.READS)),
        )

    source_cbz = workdir / "source.cbz"
    await _receive(file, source_cbz)

    chapter = name or file.filename or job_id

    _job = _record(
        job_id,
        workdir,
        kind="translate",
        name=chapter,
        filename=f"{Path(chapter).stem} [{TARGET_LANG.upper()}].cbz",
        source_lang=source,
    )

    # Detached rather than awaited: the request would otherwise stay open for
    # the whole chapter, which is the thing this API exists to avoid.
    asyncio.create_task(
        _run(_job, partial(_convert, _job, source_cbz, workdir), "translated")
    )

    return _public(_job)


@app.post("/jobs/redraw")
async def redraw(
    file: UploadFile = File(...), name: str = Form(""), document: str = Form("")
):
    """Letter a translation again, optionally from a corrected document.

    Takes a CBZ this service produced -- one carrying the cleaned pages and the
    chapter document -- and gives back the same book with stage 6 run over it
    again. Nothing is detected, cleaned, read or translated, so no model is
    loaded and a correction comes back in seconds.

    `document` is the whole chapter document with the lines edited. Sent whole
    rather than as a patch because the boxes and the measured colours in it are
    not the caller's to change, and a merge is the only other way to say so.
    Left out, the archive is simply drawn again from what it already carries.
    """
    global _job

    edited = None

    if document:
        # Parsed before the job is taken, so that a document this build cannot
        # read is a refused request rather than a job that fails a moment later.
        try:
            edited = ChapterDocument.from_dict(json.loads(document))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as problem:
            raise HTTPException(
                status_code=400, detail=f"that chapter document will not load: {problem}"
            ) from problem

    job_id, workdir = _claim()

    source_cbz = workdir / "source.cbz"
    await _receive(file, source_cbz)

    chapter = name or file.filename or job_id

    _job = _record(
        job_id,
        workdir,
        kind="redraw",
        name=chapter,
        # The same name it arrived under: a redraw replaces a book rather than
        # adding one, so renaming it would file a second copy beside the first.
        filename=Path(chapter).name or f"{job_id}.cbz",
    )

    asyncio.create_task(
        _run(_job, partial(_redraw, _job, source_cbz, workdir, edited), "redrew")
    )

    return _public(_job)


@app.get("/jobs/{job_id}/result")
async def result(job_id: str):
    if not _job or _job["id"] != job_id:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")

    if _job["state"] != "done":
        raise HTTPException(status_code=409, detail=f"job {job_id} is {_job['state']}")

    return FileResponse(
        _job["_result"],
        media_type="application/vnd.comicbook+zip",
        filename=_job["_filename"],
    )


@app.delete("/jobs/{job_id}")
async def discard(job_id: str):
    """Drop a job and its files. How a collected result gets cleaned up."""
    global _job

    if not _job or _job["id"] != job_id:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")

    if _job["state"] == "running":
        raise HTTPException(status_code=409, detail="that job is still running")

    shutil.rmtree(_job["_dir"], ignore_errors=True)
    _job = None

    return {"ok": True}


STAGE_DETAIL = {
    "clean": "erasing the original text",
    "read": "reading the dialogue",
    "translate": "translating the chapter",
    "draw": "lettering the pages",
}


async def _translate_title(conversion: FullConversion, title: str) -> str:
    """Translate the book's own title, as a request of its own.

    Not folded in with the dialogue: the chapter is sent as one ordered list so
    the translator can use what was said before and after each line, and a
    title is not part of that conversation. It is one short extra request.

    A failure here is not a failure of the translation -- the pages are done by
    this point -- so it gives up and leaves the original title instead.
    """
    if not title:
        return ""

    try:
        results = list(await conversion.translator([OcrResult(title, "")]))
    except Exception as problem:
        log.warning("could not translate the title %r: %s", title, problem)
        return ""

    return results[0].text.strip() if results else ""


def _write_pages(
    job: dict, dest: Path, names: list[str], frames: list, quality: int | None = None
) -> None:
    """Write a chapter's worth of frames into a folder, saying what would not go.

    A page that will not encode is dropped rather than fatal: the rest of the
    chapter is fine, and the alternative is losing a finished translation to one
    bad page.
    """
    dest.mkdir(parents=True, exist_ok=True)

    for name, frame in zip(names, frames):
        if not write_image(str(dest / name), frame, quality=quality):
            log.warning("job %s: could not write %s", job["id"], name)


def _convert(job: dict, source: Path, workdir: Path) -> Path:
    """Translate one CBZ, start to finish. Returns the path of the result.

    Runs on a worker thread with an event loop of its own, because the pipeline
    is only nominally async: detection, inpainting and OCR all block for
    seconds at a time, and on the main loop that would stall the very status
    endpoint the caller is polling to watch this.
    """

    def report(stage: str, done: int, total: int) -> None:
        job.update(
            stage=stage,
            done=done,
            total=total,
            detail=STAGE_DETAIL.get(stage, stage),
        )

        # How many bubbles the chapter turned out to have is only known once
        # every page has been through detection, which is what the read and
        # translate stages count over.
        if stage in ("read", "translate"):
            job["regions"] = total

    pages_dir = workdir / "pages"
    clean_dir = workdir / "clean"
    drawn_dir = workdir / "drawn"

    paths = archive.unpack(source, pages_dir)
    comicinfo = archive.read_comicinfo(source)

    loaded = [(p, read_image(str(p))) for p in paths]
    readable = [(p, frame) for p, frame in loaded if frame is not None]

    for path, frame in loaded:
        if frame is None:
            log.warning("job %s: %s could not be decoded, skipping", job["id"], path.name)

    if len(readable) == 0:
        raise ValueError("none of the pages in that archive could be read")

    job.update(pages=len(readable), stage="clean", detail=STAGE_DETAIL["clean"])

    names = [path.name for path, _ in readable]

    conversion = get_pipeline()
    # Both swapped in per job rather than fixed at construction: the pipeline
    # outlives any one job, but a progress callback and the language being read
    # both belong to one. The language picks the prompt.
    conversion.progress = report
    was_source = getattr(conversion.translator, "source_lang", "")
    if hasattr(conversion.translator, "source_lang"):
        conversion.translator.source_lang = job["source_lang"]

    async def convert():
        """Stages 1 to 6, one at a time rather than behind the whole-chapter call.

        Run out here so that what passes between the stages can be kept. The
        end-to-end call returns only the drawn pages, and the cleaned ones and
        the dialogue are exactly what a correction later needs.
        """
        layouts, ocr_results = await conversion.clean_and_read(
            [frame for _, frame in readable], names=names
        )

        document = build_document(
            job["name"], layouts, ocr_results, names=names, clean_ext=CLEAN_EXT
        )
        # What the caller said it was and what this was told to produce, rather
        # than what the OCR made of it -- the reader reports the one language
        # it knows whatever it was shown, which is no answer at all.
        document.source_language = job["source_lang"]
        document.target_language = TARGET_LANG

        # Written before stage 6 and not after, because stage 6 letters each
        # page in place: once it has run, the cleaned frame and the drawn one
        # are the same array.
        _write_pages(job, clean_dir, [Path(p.clean).name for p in document.pages],
                     [layout.frame for layout in layouts], quality=CLEAN_QUALITY)

        translations = await conversion.translate_regions(ocr_results)

        for region, translation in zip(document.regions(), translations):
            region.translation = translation.text

        frames = await conversion.render_pages(layouts, translations)

        job.update(stage="title", detail="translating the title")
        title = await _translate_title(conversion, archive.source_title(comicinfo))

        return document, frames, title

    try:
        document, frames, title = asyncio.run(convert())
    finally:
        conversion.progress = None
        if hasattr(conversion.translator, "source_lang"):
            conversion.translator.source_lang = was_source

    _write_pages(job, drawn_dir, names, frames)

    job.update(stage="pack", detail="building the archive")

    clean_zip = workdir / "clean.zip"
    archive.pack_clean(clean_dir, clean_zip)

    result_path = workdir / "result.cbz"
    archive.pack(
        drawn_dir,
        result_path,
        archive.translated_comicinfo(comicinfo, target_lang=TARGET_LANG, title=title),
        extras={archive.CLEAN_ARCHIVE: clean_zip, archive.DOCUMENT: document.to_json()},
    )

    return result_path


def _redraw(
    job: dict, source: Path, workdir: Path, edited: ChapterDocument | None
) -> Path:
    """Letter a translation again from what it carries. Returns the new archive.

    Stage 6 on its own. Everything the drawer needs -- the cleaned pages, the
    boxes, the colours measured off the original lettering, the lines -- came
    in with the archive, so nothing here loads a model or asks anything to
    translate. That is the whole reason a correction is worth offering: it
    costs seconds against the minutes the translation did.

    Runs on a worker thread for the same reason `_convert` does.
    """
    clean_dir = workdir / "clean"
    drawn_dir = workdir / "drawn"

    document = archive.read_document(source)

    if document is None:
        raise ValueError(
            "that archive carries no chapter document, so there is nothing to "
            "redraw from -- it was translated before this build kept one"
        )

    inner = archive.extract_clean_archive(source, workdir / "clean.zip")

    if inner is None:
        raise ValueError("that archive carries no cleaned pages to draw on")

    cleaned = archive.unpack_clean(inner, clean_dir)
    comicinfo = archive.read_comicinfo(source)

    # The edit wins over what was packed, but the pages are still taken from
    # the archive: a document is a description of an archive, and one that
    # named pages this archive does not have would simply draw nothing.
    if edited is not None:
        document = edited

    drawer = HorizontalDrawer()

    job.update(
        stage="draw",
        detail=STAGE_DETAIL["draw"],
        pages=len(document.pages),
        regions=len(document.regions()),
        done=0,
        total=len(document.pages),
    )

    async def draw() -> int:
        drawn_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        for page in document.pages:
            source_page = cleaned.get(Path(page.clean).name)
            frame = read_image(str(source_page)) if source_page else None

            if frame is None:
                log.warning(
                    "job %s: no cleaned page for %s, skipping", job["id"], page.name
                )
                continue

            frame = await draw_page(
                frame,
                [tuple(region.box) for region in page.regions],
                [
                    TranslatorResult(region.translation, document.target_language)
                    for region in page.regions
                ],
                drawer,
                [(region.text_color, region.background_color) for region in page.regions],
                [region.outlined for region in page.regions],
            )

            if write_image(str(drawn_dir / page.name), frame):
                written += 1
            else:
                log.warning("job %s: could not write %s", job["id"], page.name)

            job["done"] = written

        return written

    written = asyncio.run(draw())

    if written == 0:
        raise ValueError("none of the pages in that archive could be drawn")

    job.update(pages=written, stage="pack", detail="building the archive")

    result_path = workdir / "result.cbz"
    archive.pack(
        drawn_dir,
        result_path,
        comicinfo,
        # The cleaned pages go straight back in, byte for byte, alongside the
        # document as edited -- so what comes out can be corrected again. The
        # ComicInfo is carried over untouched: this is the same translation,
        # lettered again, not a new one.
        extras={archive.CLEAN_ARCHIVE: inner, archive.DOCUMENT: document.to_json()},
    )

    return result_path


async def _run(job: dict, work, verb: str) -> None:
    """Drive one job and record its outcome.

    Nothing awaits this, so a failure has nowhere to propagate to except the
    job record the caller is polling -- which is exactly where it is wanted.

    `work` blocks for as long as the job takes and runs on a worker thread for
    it; `verb` is what the finished job says it did, which is the only thing
    the two kinds differ by once they are running.
    """
    try:
        result_path = await asyncio.to_thread(work)
        job.update(
            state="done",
            stage="done",
            detail=f"{verb} {job['pages']} pages",
            finished=_now(),
            _result=result_path,
        )
        log.info("job %s finished: %s pages", job["id"], job["pages"])
    except Exception as problem:
        log.warning("job %s failed: %s", job["id"], problem, exc_info=True)
        job.update(
            state="error",
            stage="error",
            detail=str(problem),
            finished=_now(),
        )
