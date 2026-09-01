"""The parts of the pipeline that can be swapped out.

One module per stage - cleaning, ocr, translation, drawing - each holding the
backends for that stage, and one registry function per stage listing them. A new
backend is a class in the stage's module and a name in its list here; the CLI
takes it by its index in that list.

Every stage runs one real backend today. The no-ops (NoOcr, DebugTranslator)
are kept because they are how you exercise the rest of the pipeline without an
API key or a GPU.
"""

from translator.plugins.base import (
    BasePlugin,
    Cleaner,
    Drawable,
    Drawer,
    Ocr,
    OcrResult,
    Translator,
    TranslatorResult,
)
from translator.plugins.cleaning import LamaCleaner
from translator.plugins.drawing import HorizontalDrawer
from translator.plugins.ocr import JapaneseOcr, NoOcr
from translator.plugins.translation import DebugTranslator, DeepSeekTranslator


def get_cleaners() -> list[type[Cleaner]]:
    return [x for x in [LamaCleaner] if x.is_valid()]


def get_ocr() -> list[type[Ocr]]:
    return [x for x in [JapaneseOcr, NoOcr] if x.is_valid()]


def get_translators() -> list[type[Translator]]:
    return [x for x in [DeepSeekTranslator, DebugTranslator] if x.is_valid()]


def get_drawers() -> list[type[Drawer]]:
    return [x for x in [HorizontalDrawer] if x.is_valid()]
