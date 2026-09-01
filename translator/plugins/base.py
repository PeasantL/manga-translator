import numpy as np
from typing import Union

class BasePlugin:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_name() -> str:
        return "unknown"

    @staticmethod
    def is_valid() -> bool:
        return True


class OcrResult:
    def __init__(self, text: str = "", language: str = "en") -> None:
        self.text = text
        self.language = language


class Ocr(BasePlugin):
    """Always outputs \"Sample\" """

    def __init__(self) -> None:
        super().__init__()

    async def __call__(self, batch: list[np.ndarray]) -> list[OcrResult]:
        return await self.do_ocr(batch)

    async def do_ocr(self, batch: list[np.ndarray]):
        return [OcrResult("Sample") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Ocr"


class TranslatorResult:
    def __init__(self, text: str = "", lang_code: str = "en") -> None:
        self.lang_code = lang_code
        self.text = text


class Translator(BasePlugin):
    """Base Class for all Translator classes"""

    def __init__(self) -> None:
        super().__init__()

    async def __call__(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return await self.translate(batch)

    async def translate(self, batch: list[OcrResult]) -> list[TranslatorResult]:
        return [TranslatorResult(x.text) for x in batch]

    @staticmethod
    def get_name() -> str:
        return "Base Translator"

class Drawable:
    def __init__(
        self,
        color: tuple[np.ndarray, np.ndarray, bool],
        translation: TranslatorResult,
        frame: np.ndarray,
        backdrop: bool = False,
        page_shape: Union[tuple, None] = None,
    ) -> None:
        self.color = color
        self.translation = translation
        self.frame = frame
        # Set when the text did not fit its bubble and is being drawn over
        # artwork instead, so the drawer knows to put something behind it.
        self.backdrop = backdrop
        # The whole page this region was cut out of. A drawer needs it because
        # a readable size is a fraction of the page rather than a count of
        # pixels: the same chapter is scanned at anything from 800 to 2400
        # pixels high, and lettering has to follow that or it is fine print on
        # one scan and a headline on the next.
        self.page_shape = page_shape

class Drawer(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    def box_for(
        self,
        text: str,
        box: tuple[int, int, int, int],
        page_shape: tuple,
    ) -> tuple[tuple[int, int, int, int], bool]:
        """The area this drawer needs for `text`, and whether it grew.

        Text layout belongs to the drawer, so the pipeline asks rather than
        assuming the detected box is usable. The base drawer draws nothing, so
        it always takes what it is given.
        """
        return tuple(int(v) for v in box), False


    async def draw(
        self, batch: list[Drawable]
    ) -> list[tuple[np.ndarray,np.ndarray]]:
        # An empty mask means "draw nothing", which is what this no-op base does.
        return [
            (x.frame, np.zeros(x.frame.shape[:2], dtype=np.uint8)) for x in batch
        ]

    async def __call__(
        self, batch: list[Drawable]
    ) -> list[tuple[np.ndarray,np.ndarray]]:
        return await self.draw(batch=batch)


class Cleaner(BasePlugin):
    def __init__(self) -> None:
        super().__init__()

    async def clean(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[np.ndarray, np.ndarray]:
        return frame, mask

    async def __call__(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[np.ndarray, np.ndarray]:
        return await self.clean(frame=frame, mask=mask, detection_results=detection_results)
