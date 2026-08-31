# input

Put a chapter's pages in here, then run either of:

```bash
python3 main.py -f input
./run.sh
```

`run.sh` does the same thing after creating the virtualenv and installing
dependencies, if you have not already.

A folder is treated as **one chapter**. Every page is detected and cleaned
first, then the chapter's dialogue is read and translated as a single ordered
list, so the translator can use the surrounding lines for context. Pass a
chapter at a time rather than a page at a time.

Pages are sorted naturally, so `page2` comes before `page10`. Name them so they
sort into reading order — that order is what the translator sees.

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`,
`.tiff`. Anything else in here is skipped with a message rather than failing
the run, which is why this file being here is harmless.

Converted pages are written to [`../output`](../output).

Everything you put in this folder is gitignored except this file.
