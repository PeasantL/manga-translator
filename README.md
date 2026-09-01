# Manga Translator

Detects speech bubbles on a manga page, erases the original text, translates it,
and draws the translation back into the bubble.

Forked from [TareHimself/manga-translator](https://github.com/TareHimself/manga-translator).

## How it works

A folder is one chapter. It runs through six stages, in
`translator/pipeline.py`:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) finds speech bubbles and free text |
| 2 | Segmentation | A second YOLO model masks the text pixels inside them |
| 3 | Cleaning | [LaMa](https://github.com/advimman/lama) inpaints the text away |
| 4 | OCR | [manga-ocr](https://huggingface.co/TareHimself/manga-ocr-base) reads every bubble in the chapter, in reading order |
| 5 | Translation | DeepSeek translates that whole list in one request |
| 6 | Drawing | PIL lays the translation out and draws it into the cleaned bubble |

Stages 1 to 3 and stage 6 run per page. Stages 4 and 5 run once for the whole
chapter: every page is detected and cleaned first, then the chapter's dialogue is
read and translated as a single ordered list, and only then is anything drawn.
Translating a bubble on its own gives the model no idea who is speaking or what
was just said, so lines are ordered the way the page is read — rows top to
bottom, right to left within a row — and sent together.

Chapters longer than the translator's `max_lines` (200 by default) are split
across requests, with the previous lines carried over as context.

### The three stages

The six steps are grouped into three stages that can each be run on their own,
with JSON between them:

| Stage | Steps | Reads | Writes |
|-------|-------|-------|--------|
| `ocr` | 1–4 | the chapter's pages | `clean/` and `ocr.json` |
| `translate` | 5 | `ocr.json` | `translated.json` |
| `draw` | 6 | `translated.json` and `clean/` | `drawn/` |

Each stage builds only what it needs. `translate` and `draw` load no detection,
cleaning or OCR model at all, so re-running them costs seconds rather than
minutes — and `ocr` needs no API key.

`ocr.json` holds, for every region, the box it will be drawn into, the text read
out of it, its language, and the colour of the lettering and of what the
lettering sat on. `translated.json` is the same document with a translation
filled in per region. Because all of that travels with the text, the draw stage
never has to detect anything or look at the original page again - a white on
black bubble is lettered white because that is what was measured off it, not
because anything guessed. Both are ordinary JSON: correct a translation by hand,
re-run `-s draw`, and only the drawing happens again.

Stages 3 to 6 are plugins, so adding a backend means writing one class and
adding it to that stage's list in `translator/plugins/__init__.py`. The CLI
takes a backend by its index in that list.

Each stage deliberately carries one good backend rather than a menu. The
alternatives that used to be here — EasyOCR, Tesseract, DeepFillV2, OpenAI,
Gemini, Google Cloud, Helsinki-NLP, DeepL — were either worse on manga specifically,
broken, or duplicated something that remains.

## Install

Requires Python 3.10 or newer.

### 1. System packages

None are required. Optionally, `sudo apt install python3-tk` enables the debug
image viewer; without it, `display_image()` writes a PNG to the working
directory instead.

On Ubuntu, installing
[Intel oneAPI Threading Building Blocks](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb)
improves CPU inference performance.

### 2. Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the torch stack from the PyTorch index first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Model weights

The weights are not in the repository. Download them into `models/`:

```bash
./fetch_models.sh
```

That fetches the two required models (~258 MB):

| File | Size | Used by |
|------|------|---------|
| `detection.pt` | 50 MB | Bubble and free-text detection |
| `segmentation.pt` | 208 MB | Text segmentation |

The LaMa cleaner and manga-ocr download their own weights the first time they
run, so they are not listed here.

## Usage

Commands are run from the repository root.

### CLI

`-f` takes chapters. Each folder you pass is one chapter, and gets a folder of
the same name under `output/`:

```
input/my-oneshot/             output/my-oneshot/
    01.png                        clean/01.png     cleaned, no text
    02.png                        clean/02.png
                                  drawn/01.png     finished pages
                                  drawn/02.png
                                  ocr.json         boxes and source text
                                  translated.json  the same, translated
```

```bash
python3 main.py -f input/my-oneshot                   # every stage
python3 main.py -f input/my-oneshot -s ocr            # or one stage at a time
python3 main.py -f input/my-oneshot -s translate
python3 main.py -f input/my-oneshot -s draw
python3 main.py -f input/one input/two                # several chapters at once
```

Pass the chapter, not the folder your chapters live in — pointing `-f` at
`input` would make `input` itself the chapter. Pages are sorted naturally, so
`page2` comes before `page10` — name them so they sort into reading order,
because that order is what the translator sees. `python3 main.py --help` lists the available
OCR, translator, drawer and cleaner backends with their index numbers, which is
what `-o`, `-t`, `-dr` and `-c` take. Each has a matching `-oa`, `-ta`, `-dra`
and `-ca` for that backend's settings, e.g. `-ca "dilation=15"` to erase more
aggressively.

`./run.sh` is a shortcut that creates the venv if needed, installs
dependencies, and converts every chapter folder in `./input`.

### As a service

`service.py` is the other way in, and the one anything other than a person at a
terminal uses. It takes a whole chapter as a CBZ and hands back a translated
CBZ — which takes minutes, far too long to hold a request open for, so
submitting one starts a job:

```bash
docker compose up -d                                    # or: uvicorn service:app --port 1007

curl -X POST localhost:1007/jobs -F file=@chapter.cbz \
     -F source_lang=zh                                  # -> {"id": "...", "state": "running"}
curl localhost:1007/jobs/current                        # poll: stage, done/total
curl -o out.cbz localhost:1007/jobs/<id>/result         # once state is "done"
curl -X DELETE localhost:1007/jobs/<id>                 # collected, drop it
```

| Route | Does |
|-------|------|
| `POST /jobs` | multipart `file` (a CBZ), optional `name` and `source_lang`. 409 while one is running |
| `GET /jobs/current` | the job record, or `{"state": "idle"}` |
| `GET /jobs/{id}` | the same, by id |
| `GET /jobs/{id}/result` | the translated CBZ. 409 unless the job is done |
| `DELETE /jobs/{id}` | drop the job and its files |
| `GET /healthz` | device, whether the weights are present, whether it is busy, and which sources the OCR `reads` |

One job runs at a time: there is one GPU and the models are process-global, so a
second chapter running alongside would only make both slower.

This process holds the job, and goes on holding it after it finishes. Whoever
submitted a chapter can go away, restart, and come back to find the finished
file still waiting — which is what lets the caller keep no state of its own. The
conversion runs on a worker thread, because detection, inpainting and OCR each
block for seconds at a time and on the main loop they would stall the very
endpoint the caller is polling to watch them.

Pages are extracted under sequential names rather than the archive's own. An
archive's names sort lexically wherever something else reads them, so `10.webp`
lands before `2.webp` and the chapter comes out in the wrong order; and an entry
name is written by whoever packed the file, so `../../etc/passwd` is a valid
one. Renaming on the way in settles both.

#### What it reads

`source_lang` on a job says what the pages are in, and it picks the prompt the
chapter is translated with. There is one per source rather than one prompt
naming a language, because what a translator needs telling differs by more than
the language does:

| Source | The prompt asks for |
|---|---|
| `ja` | Japanese given-name order, Hepburn romanisation, honorifics kept attached to names, kana sound effects as English sound words |
| `zh` | pinyin without tone marks and surname first, Chinese forms of address rather than invented Japanese honorifics, chengyu as English idiom, no pinyin left in the output |

Anything else falls back to a generic comics prompt rather than failing.

The caller sends it because this process cannot work it out: the OCR reports
the one language it was trained on whatever it was given. `SOURCE_LANG` is only
the default for a caller that says nothing.

> **The OCR reads Japanese only.** manga-ocr is trained on Japanese manga and
> turns anything else into plausible-looking Japanese nonsense — which is worse
> than an empty bubble, because it then translates cleanly. The prompts and the
> plumbing around them are in place ahead of a reader that can use them; a
> `zh` job today gets the right prompt applied to whatever manga-ocr made of
> the text, and logs a warning saying so. `GET /healthz` reports what the OCR
> actually `reads`, which is what a caller should check.

The source's `ComicInfo.xml` is carried across with its artist, tags and origin
intact — the translation is the same book — and only what is no longer true
changes: `LanguageISO`, a `translated` tag, and ` [EN]` on the title. Running it
over an already-translated file adds none of them twice.

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `1007` | Port to listen on |
| `TARGET_LANG` | `en` | What to translate into |
| `SOURCE_LANG` | `ja` | What to read a chapter as when the caller does not say. Picks the prompt |
| `TRANSLATOR` | `deepseek` | `debug` letters a fixed string instead, and needs no API key |
| `JOB_DIR` | `jobs` | Scratch for the chapter in flight. Cleared at startup |
| `MODELS_DIR` | `models` | Where `fetch_models.sh` puts the YOLO weights |

### API keys

The translation backend needs credentials. Copy the template and fill it in:

```bash
cp .env.example .env      # then set DEEPSEEK_API_KEY
```

One file covers both ways of running this. `main.py` and `service.py` load it
directly, and `docker compose` reads the same file to fill in
`docker-compose.yml`. It is gitignored; `.env.example` is the copy that
is checked in.

Only translating needs the key. Detection, cleaning, OCR and lettering all run
without one, so `-s ocr` works on an empty key — as does the debug translator,
which writes a fixed string into every bubble and is how to check those stages
on their own. The service reaches it as `TRANSLATOR=debug`.

## Datasets

The detection and segmentation models were trained on these. The training
scripts and dataset converters are not in this repository — see the upstream
project if you want to retrain.

- [Detection](https://universe.roboflow.com/tarehimself/manga-translator-detection)
- [Segmentation](https://universe.roboflow.com/tarehimself/manga-translator-segmentation)

## Examples

`examples/raw` holds four sample pages, which is one chapter:

```bash
python3 main.py -f examples/raw
```

The converted pages are written to `output/raw/drawn/`. They are not checked in, since
the result depends on which model and target language you translate with.

## Layout

```
main.py                     the CLI
service.py                  the job API: a CBZ in, a translated CBZ out
run.sh                      venv, dependencies, then every chapter in input/
fetch_models.sh             downloads the YOLO weights into models/
Dockerfile                  CUDA image for the service
docker-compose.yml          the service, with the weights and caches as volumes
translator/
    chapter.py              what a chapter is, and the JSON passed between stages
    archive.py              CBZ to chapter folder and back, and ComicInfo
    pipeline.py             the six stages, wired together
    utils.py                image, text layout and colour helpers
    plugins/
        base.py             what a backend has to implement
        __init__.py         the registry each stage's backends are listed in
        cleaning.py         stage 3
        ocr.py              stage 4
        translation.py      stage 5
        drawing.py          stage 6
input/  output/  fonts/  models/  examples/
```

## Glossary

- **Bubble**: a speech bubble
- **Free text**: text found on pages but not in speech bubbles
- **Bubble text**: text within speech bubbles

## Status

Working:

- Bubble and free text detection
- Bubble text extraction, masking and inpainting
- OCR, translation, hyphenation and text insertion

Not working yet:

- Vertical text drawing
- Free text OCR and translation quality
- Text resize algorithm — some text comes out too large or too small

Removed rather than fixed: text colour detection (`translator/color_detect/`)
and the `VerticalDrawer` stub. Both were dead code; the git history has them if
either is ever picked back up.

## License

GPL-3.0. See [LICENSE](LICENSE).
