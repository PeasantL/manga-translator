# Manga Translator

Detects speech bubbles on a manga page, erases the original text, translates it,
and draws the translation back into the bubble.

Forked from [TareHimself/manga-translator](https://github.com/TareHimself/manga-translator).

## How it works

A folder is treated as one chapter. It runs through six stages, in
`translator/pipelines.py`:

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

Stages 3 to 6 are plugins. Each declares its own settings, and the web UI builds
its settings form from those declarations, so adding a backend means writing one
class and adding one line to the matching `get.py`.

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
./scripts/fetch_models.sh
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

`-f` takes the folder that holds your chapters. Each folder inside it is one
chapter, and gets a folder of the same name under `output/`:

```
input/                        output/
    my-oneshot/                   my-oneshot/
        01.png                        clean/01.png     cleaned, no text
        02.png                        clean/02.png
                                      drawn/01.png     finished pages
                                      drawn/02.png
                                      ocr.json         boxes and source text
                                      translated.json  the same, translated
```

```bash
python3 main.py -f input                   # every chapter, every stage
python3 main.py -f input -s ocr            # or one stage at a time
python3 main.py -f input -s translate
python3 main.py -f input -s draw
```

A chapter needs a folder. Images lying loose in the input root are skipped,
since there would be no name to give their output folder. Pages are sorted
naturally, so `page2` comes before `page10` — name them so they sort into
reading order, because that order is what the translator sees. `python3 main.py --help` lists the available
OCR, translator, drawer and cleaner backends with their index numbers, which is
what `-o`, `-t`, `-dr` and `-c` take. Each has a matching `-oa`, `-ta`, `-dra`
and `-ca` for that backend's settings, e.g. `-ca "dilation=15"` to erase more
aggressively.

`./run.sh` is a shortcut that creates the venv if needed, installs
dependencies, and converts everything in `./input`.

### Web UI

```bash
python3 server.py
```

Serves the interface on <http://localhost:5000>, where you can pick backends,
set their arguments, and compare the original against the result.

The UI is a React app in `ui/`. To rebuild it after changing the source:

```bash
cd ui && npm install && npm run build
```

### API keys

The translation backend needs credentials. The web UI has a field for the key;
`server.py` and `main.py` both read a `.env` file in the repository root:

```
DEEPSEEK_API_KEY=...
```

The `Custom Text` translator writes a fixed string into every bubble and needs
no key — useful for checking detection, cleaning and drawing on their own.

## Datasets

The detection and segmentation models were trained on these. The training
scripts and dataset converters are not in this repository — see the upstream
project if you want to retrain.

- [Detection](https://universe.roboflow.com/tarehimself/manga-translator-detection)
- [Segmentation](https://universe.roboflow.com/tarehimself/manga-translator-segmentation)

## Examples

`examples/raw` holds six sample pages. Pointing `-f` at `examples` treats
`raw` as a chapter:

```bash
python3 main.py -f examples
```

The converted pages are written to `output/raw/drawn/`. They are not checked in, since
the result depends on which model and target language you translate with.

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
