from translator.core.plugin import Ocr
from translator.ocr.no import NoOcr
from translator.ocr.huggingface_ja import JapaneseOcr


def get_ocr() -> list[Ocr]:
    return list(filter(lambda a: a.is_valid(), [JapaneseOcr, NoOcr]))
