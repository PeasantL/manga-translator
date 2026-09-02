# MangaTranslate

Detects speech bubbles on a manga page, erases the original text, translates it,
and draws the translation back into the bubble.

Forked from [TareHimself/manga-translator](https://github.com/TareHimself/manga-translator).

> **Archived.** This is no longer developed on its own. It now lives as a
> subdirectory of the manga reader it was written for, vendored there with
> `git subtree`: nothing ran it except that reader, and the two were coupled far
> more tightly than two separate repositories admitted. The reader parses a
> document this writes *inside* the translated archive, under `translator/`, to
> let a person correct the lines — and nothing on either side checked that the
> two still agreed about its shape.
>
> What is here is the final standalone state, and it works: `./run.sh` for the
> CLI, `service.py` for the HTTP job API, `./fetch_models.sh` for the weights.
> The reader is a private repository, so this stays the public copy of the
> source.

## How it works

A folder is one chapter. It runs through six stages, in
`translator/pipeline.py`:

| # | Stage | Model | What it does |
|---|-------|-------|--------------|
| 1 | Detection | [comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | RT-DETR-v2 finds the balloons, the text inside them, and text in no balloon |
| 2 | Segmentation | [comic-text-segmenter](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) | YOLOv8m marks the text pixels themselves, which is what gets erased |
| 3 | Cleaning | [AnimeMangaInpainting](https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript) | [LaMa](https://github.com/advimman/lama), finetuned on anime and manga, paints the text away |
| 4 | OCR | [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) | A 1B vision language model reads every region in the chapter, in reading order |
| 5 | Translation | `deepseek-v4-pro` | DeepSeek translates that whole list in one request |
| 6 | Drawing | — | PIL lays the translation out and draws it into the cleaned bubble |

Stage 1 tells the balloon apart from the lettering in it, which is what lets each
half go where it is useful: the reader is handed the lettering alone, and the
letterer is handed the whole balloon to fit a translation into. Text the detector
finds outside a balloon is erased only where the segmenter agrees it is
lettering — otherwise a chapter title or a logo, which this detector does find,
would be painted out along with the artwork behind it.

Stages 1 to 3 and stage 6 run per page. Stages 4 and 5 run once for the whole
chapter: every page is detected and cleaned first, then the chapter's dialogue is
read and translated as a single ordered list, and only then is anything drawn.
Translating a bubble on its own gives the model no idea who is speaking or what
was just said, so lines are ordered the way the page is read — rows top to
bottom, right to left within a row — and sent together.

Chapters longer than the translator's `max_lines` (200 by default) are split
across requests, with the previous lines carried over as context.

The OCR reads Japanese best — that is what it was finetuned on — and Chinese as
well as the model it was finetuned from did, which is less well. Everything
before it is trained on manga, manhua, webtoons and western comics alike.

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

### Lettering

Stage 6 draws each translation at the largest size it fits its bubble at,
between a minimum and a maximum. Both are quoted for a 1200 pixel page and
scaled to the page actually being drawn on: a bubble covers the same fraction of
the page however finely it was scanned, so as literal pixel counts one setting
is fine print on a 2000 pixel scan and a headline on an 800 pixel one.

| Setting | Default | Meaning |
|---|---|---|
| `min_font_size` | `11` | Below this the text spills out of the bubble rather than shrinking further |
| `max_font_size` | `30` | The largest a short line in a roomy bubble is drawn |
| `line_spacing` | `2` | Leading between lines |

Pass them with `-dra`, e.g. `-dra "max_font_size=36"`. On the 1600 pixel example
pages those defaults come out as 15 and 40.

A translation that does not fit at the minimum grows its box instead, up to
2.5x, and is drawn over the artwork with an outline — or on a panel of its own
where what is underneath is too solid to read against. The room it grows into is
whatever the other bubbles on the page are not using, and a box that started out
much taller than it is wide — a strip of vertical Japanese — is squared up as it
grows rather than scaled, because English set in a column comes out one word to
a line.

Words are broken across lines only where a dictionary says they may, and only
with a hyphen: a break that would leave one or two letters stranded is not taken,
and where a size a point or two smaller sets every word whole, that size wins.

Stages 3 to 6 are plugins, so adding a backend means writing one class and
adding it to that stage's list in `translator/plugins/__init__.py`. The CLI
takes a backend by its index in that list.

Each stage deliberately carries one good backend rather than a menu. The
alternatives that used to be here — EasyOCR, Tesseract, DeepFillV2, OpenAI,
Gemini, Google Cloud, Helsinki-NLP, DeepL — were either worse on manga specifically,
broken, or duplicated something that remains. The backends that stage 1 to 4
carry today replaced a set trained in 2022 and 2023: a pair of YOLOv8 models
that knew manga only, the generic photographic LaMa, and manga-ocr.

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

Nothing has to be downloaded by hand: each of the four models is fetched from
the Hugging Face hub the first time it is used and cached in `HF_HOME` — a
volume in the compose file, so a rebuild does not fetch them again. About 5 GB
in total, most of it the OCR model.

To get the download over with before a chapter is waiting on it:

```bash
./fetch_models.sh
```

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
| `POST /jobs/redraw` | multipart `file` (a CBZ this made), optional `name` and `document`. Letters it again |
| `GET /jobs/current` | the job record, or `{"state": "idle"}` |
| `GET /jobs/{id}` | the same, by id |
| `GET /jobs/{id}/result` | the finished CBZ. 409 unless the job is done |
| `DELETE /jobs/{id}` | drop the job and its files |
| `GET /healthz` | device, whether the weights are present, whether it is busy, and which sources the OCR `reads` |

One job runs at a time: there is one GPU and the models are process-global, so a
second chapter running alongside would only make both slower.

The result's ComicInfo says it is a translation twice, and those are not
duplicates of each other. `Tags` gets `translated` and `Genre` gets
`Translated`, because a library reading ComicInfo files the two in different
places — Tags describe the book, Genre describes the work, so in Komga the tag
lands on the book record and the genre on the series wrapping it. Which one is
visible depends on where you are looking from, and a shelf of covers is series.

#### Correcting a translation

A translated CBZ carries what it took to make it, under `translator/` inside the
archive: `chapter.json` is the chapter document — every bubble, where it is, what
it said and what that became — and `clean.zip` holds the pages with the original
text erased. Together they are exactly what stage 6 needs, so a wrong line can be
fixed without the chapter going near a model again, and without the untranslated
book still having to be there.

```bash
unzip -p out.cbz translator/chapter.json > lines.json   # edit the translations
curl -X POST localhost:1007/jobs/redraw -F file=@out.cbz \
     -F document=@lines.json                            # -> a job, as above
```

`document` is the whole chapter document with the lines edited, not a patch: the
boxes and the colours measured off the original lettering are not the caller's to
change, and a merge is the only other way to say so. Left out, the archive is
simply drawn again from what it already carries.

Nothing is detected, cleaned, read or translated, so no model is loaded and a
correction comes back in seconds rather than minutes. What comes out carries the
sidecar too, so it can be corrected again.

The cleaned pages are WebP rather than the lossless PNGs the CLI leaves in
`output/`: these go into somebody's library, and lossless would roughly double
what a chapter costs to keep. Nothing accumulates from it, because a redraw
always starts from the same cleaned page. `CLEAN_QUALITY` above 100 buys lossless
back.

They are nested in an archive of their own for a reason worth knowing before
moving them: a comic reader decides what the pages of a CBZ are by walking its
entries for images at any depth, so cleaned pages sitting loose in there would
double every chapter's page count. Nothing descends into a nested zip.

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

The caller sends it because it is the one that knows: the OCR reports the script
it read rather than the language a chapter is in, and kanji alone are not enough
to tell Japanese from Chinese. `SOURCE_LANG` is the default for a caller that
says nothing.

> **The OCR reads Japanese and Chinese**, and Japanese much better — the model
> was finetuned on Japanese manga, and its Chinese is what the base model came
> with. A job in anything else gets the right prompt applied to whatever the
> reader made of the text, and logs a warning saying so. `GET /healthz` reports
> what the OCR actually `reads`, which is what a caller should check.

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
| `HF_HOME` | | Where the model weights are cached. `/cache/huggingface` in the container |

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
fetch_models.sh             pre-fetches every model's weights into the cache
Dockerfile                  CUDA image for the service
docker-compose.yml          the service, with the weight cache and jobs as volumes
translator/
    chapter.py              what a chapter is, and the JSON passed between stages
    archive.py              CBZ to chapter folder and back, and ComicInfo
    pipeline.py             the six stages, wired together; stages 1 and 2 live here
    utils.py                image, text layout and colour helpers
    plugins/
        base.py             what a backend has to implement
        __init__.py         the registry each stage's backends are listed in
        cleaning.py         stage 3
        ocr.py              stage 4
        translation.py      stage 5
        drawing.py          stage 6
input/  output/  fonts/  examples/
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

Removed rather than fixed: text colour detection (`translator/color_detect/`)
and the `VerticalDrawer` stub. Both were dead code; the git history has them if
either is ever picked back up.

## License

GPL-3.0. See [LICENSE](LICENSE).
