import numpy
from translator.utils import cv2_to_pil, lang_code_to_name
from translator.core.plugin import (
    Ocr,
    OcrResult,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)


class EasyOcr(Ocr):
    """Supports all the languages listed"""

    languages = [
        "ja",
        "abq",
        "ady",
        "af",
        "ang",
        "ar",
        "as",
        "ava",
        "az",
        "be",
        "bg",
        "bh",
        "bho",
        "bn",
        "bs",
        "ch_sim",
        "ch_tra",
        "che",
        "cs",
        "cy",
        "da",
        "dar",
        "de",
        "en",
        "es",
        "et",
        "fa",
        "fr",
        "ga",
        "gom",
        "hi",
        "hr",
        "hu",
        "id",
        "inh",
        "is",
        "it",
        "kbd",
        "kn",
        "ko",
        "ku",
        "la",
        "lbe",
        "lez",
        "lt",
        "lv",
        "mah",
        "mai",
        "mi",
        "mn",
        "mr",
        "ms",
        "mt",
        "ne",
        "new",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "ru",
        "rs_cyrillic",
        "rs_latin",
        "sck",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "ta",
        "tab",
        "te",
        "th",
        "tjk",
        "tl",
        "tr",
        "ug",
        "uk",
        "ur",
        "uz",
        "vi",
    ]

    def __init__(self, lang=languages[0]) -> None:
        import easyocr

        super().__init__()
        self.easy = easyocr.Reader([lang])
        self.language = lang

    async def do_ocr(self, batch: list[numpy.ndarray]):
        result = []
        for x in batch:
            # readtext(paragraph=True) groups the detected words into paragraphs
            # and returns a list of them. Taking [0] dropped everything after
            # the first group, so any bubble easyocr split in two lost half its
            # text with no warning.
            paragraphs = self.easy.readtext(x, detail=0, paragraph=True)
            text = " ".join(paragraphs)
            result.append(OcrResult(text=text, language=self.language))
        return result

    @staticmethod
    def get_name() -> str:
        return "Easy Ocr"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        options = list(
            filter(
                lambda a: a.name is not None,
                [
                    PluginSelectArgumentOption(name=lang_code_to_name(lang), value=lang)
                    for lang in EasyOcr.languages
                ],
            )
        )

        return [
            PluginSelectArgument(
                id="lang",
                name="Language",
                description="The language to detect",
                options=options,
                default=options[0].value,
            )
        ]
