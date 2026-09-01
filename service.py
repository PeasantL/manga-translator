"""A job API over the pipeline: one CBZ in, one translated CBZ out.

server.py is the interactive way in -- one image, one request, a person looking
at the result. This is the other way: a library hands over a whole chapter and
collects a translated one back minutes later. That is far too long to hold a
request open for, so submitting a chapter starts a job and returns immediately.

This process is the one that holds the job. Whoever submitted a chapter can go
away, restart, and come back to find the finished file still waiting, which is
what lets the caller keep no state of its own. A finished job is kept until it
is collected and deleted.

One job runs at a time. There is one GPU, the models are process-global, and a
second chapter running concurrently would only make both slower.
"""

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from translator import archive
from translator.pipeline import FullConversion
from translator.plugins import DebugTranslator, DeepSeekTranslator, JapaneseOcr, OcrResult
from translator.utils import read_image, write_image

log = logging.getLogger("translator.service")

JOB_DIR = Path(os.environ.get("JOB_DIR", "jobs"))
TARGET_LANG = os.environ.get("TARGET_LANG", "en")

# "debug" writes a fixed string into every bubble and needs no API key, which
# is how detection, cleaning and lettering get exercised on their own -- worth
# having as a setting because it is also how this service is smoke tested.
TRANSLATORS = {"deepseek": DeepSeekTranslator, "debug": DebugTranslator}
TRANSLATOR = os.environ.get("TRANSLATOR", "deepseek").lower()

# Read in chunks rather than with .read(): a chapter is tens of megabytes and
# there is no reason for all of it to be resident at once.
UPLOAD_CHUNK = 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any job directory still here belongs to a previous process, whose record
    # of it died with it. Nothing can collect those, so they are just disk.
    if JOB_DIR.exists():
        for stale in JOB_DIR.iterdir():
            shutil.rmtree(stale, ignore_errors=True)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="manga-translator", lifespan=lifespan)

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
        _pipeline = FullConversion(ocr=JapaneseOcr(), translator=translator)

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
    health check never has the side effect of pulling a gigabyte into memory.
    """
    import torch

    models = Path(os.environ.get("MODELS_DIR", "models"))

    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": all((models / m).is_file() for m in ("detection.pt", "segmentation.pt")),
        "loaded": _pipeline is not None,
        "busy": _job is not None and _job["state"] == "running",
        "target_lang": TARGET_LANG,
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


@app.post("/jobs")
async def submit(file: UploadFile = File(...), name: str = Form("")):
    """Take a CBZ and start translating it.

    A finished job that nobody collected is replaced rather than protected: the
    caller polls, and one that has stopped polling long enough to submit
    something else is not coming back for it.
    """
    global _job

    if _job and _job["state"] == "running":
        raise HTTPException(status_code=409, detail="a translation is already running")

    if _job:
        shutil.rmtree(_job["_dir"], ignore_errors=True)

    job_id = uuid.uuid4().hex
    workdir = JOB_DIR / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "source.cbz"

    try:
        with source.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                out.write(chunk)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise

    _job = {
        "id": job_id,
        "name": name or file.filename or job_id,
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
    }

    # Detached rather than awaited: the request would otherwise stay open for
    # the whole chapter, which is the thing this API exists to avoid.
    asyncio.create_task(_run(_job, source, workdir))

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
        filename=f"{Path(_job['name']).stem} [{TARGET_LANG.upper()}].cbz",
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

    conversion = get_pipeline()
    # Swapped in per job rather than fixed at construction, because the
    # pipeline outlives any one job but a progress callback belongs to one.
    conversion.progress = report

    async def convert() -> tuple[list, str]:
        pages = await conversion([frame for _, frame in readable])
        job.update(stage="title", detail="translating the title")
        return pages, await _translate_title(conversion, archive.source_title(comicinfo))

    try:
        frames, title = asyncio.run(convert())
    finally:
        conversion.progress = None

    drawn_dir.mkdir(parents=True, exist_ok=True)

    for (path, _), frame in zip(readable, frames):
        if not write_image(str(drawn_dir / path.name), frame):
            log.warning("job %s: could not write %s", job["id"], path.name)

    job.update(stage="pack", detail="building the archive")

    result_path = workdir / "result.cbz"
    archive.pack(
        drawn_dir,
        result_path,
        archive.translated_comicinfo(comicinfo, target_lang=TARGET_LANG, title=title),
    )

    return result_path


async def _run(job: dict, source: Path, workdir: Path) -> None:
    """Drive one job and record its outcome.

    Nothing awaits this, so a failure has nowhere to propagate to except the
    job record the caller is polling -- which is exactly where it is wanted.
    """
    try:
        result_path = await asyncio.to_thread(_convert, job, source, workdir)
        job.update(
            state="done",
            stage="done",
            detail=f"translated {job['pages']} pages",
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
