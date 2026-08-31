# input

Each chapter goes in **its own folder** in here, and you pass **that folder**,
not this one:

```
input/my-oneshot/01.png  ->  output/my-oneshot/drawn/01.png
```

```bash
python3 main.py -f input/my-oneshot
./run.sh
```

`run.sh` converts every folder in here, after creating the virtualenv and
installing dependencies if you have not already.

Each folder is **one chapter**. Every page is detected and cleaned
first, then the chapter's dialogue is read and translated as a single ordered
list, so the translator can use the surrounding lines for context. Pass a
chapter at a time rather than a page at a time.

Pages are sorted naturally, so `page2` comes before `page10`. Name them so they
sort into reading order — that order is what the translator sees.

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`,
`.tiff`. Anything else in here is skipped with a message rather than failing
the run, which is why this file being here is harmless.

Results go to `../output/<folder name>/`: `drawn/` (the finished pages),
`clean/` (the pages with the text erased), `ocr.json` (each bubble's box and
source text) and `translated.json` (the same with translations). Correct a
translation in that file by hand and re-run `python3 main.py -f input -s draw`
to redraw without detecting, reading or translating again.

Everything you put in this folder is gitignored except this file.
