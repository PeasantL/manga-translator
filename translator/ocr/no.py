import numpy
from translator.core.plugin import Ocr, OcrResult


class NoOcr(Ocr):
    """Skips OCR. Use it to clean a page without translating it"""

    def __init__(self) -> None:
        super().__init__()

    async def do_ocr(self, batch: list[numpy.ndarray]):
        return [OcrResult("", "") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "No Ocr"
