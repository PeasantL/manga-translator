import asyncio
import os
from translator.utils import get_languages
from translator.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    PluginSelectArgument,
    PluginSelectArgumentOption,
    PluginTextArgument,
    PluginArgument,
)


class DeepSeekTranslator(Translator):
    """Translates using the DeepSeek API"""

    # DeepSeek serves an OpenAI-compatible API, so the openai client works
    # against it once the base url is pointed here.
    BASE_URL = "https://api.deepseek.com"

    MODELS = [
        ("DeepSeek Chat", "deepseek-chat"),
        ("DeepSeek Reasoner", "deepseek-reasoner"),
    ]

    def __init__(
        self, api_key="", target_lang="en", model=MODELS[0][1], temp="1.3"
    ) -> None:
        super().__init__()
        from openai import AsyncOpenAI

        # The UI sends the key as an argument; fall back to the environment so
        # a .env file works for the CLI.
        key = api_key.strip() or os.getenv("DEEPSEEK_API_KEY", "")

        self.client = (
            AsyncOpenAI(api_key=key, base_url=DeepSeekTranslator.BASE_URL)
            if key
            else None
        )
        self.target_lang = target_lang
        self.model = model
        self.temp = float(temp)

    async def translate_one(self, ocr_result: OcrResult):
        if len(ocr_result.text.strip()) == 0:
            return TranslatorResult("", self.target_lang)

        message = (
            f"{ocr_result.language.upper()} to {self.target_lang.upper()}\n"
            f"{ocr_result.text}"
        )

        result = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {
                    "role": "system",
                    "content": "You translate manga dialogue. Reply with the "
                    "translation only, no commentary and no quotes.",
                },
                {"role": "user", "content": "EN to JA\nHello"},
                {"role": "assistant", "content": "こんにちは"},
                {"role": "user", "content": message},
            ],
        )

        return TranslatorResult(
            result.choices[0].message.content.strip(), self.target_lang
        )

    async def translate(self, batch: list[OcrResult]):
        if self.client is None:
            return [TranslatorResult("Need DeepSeek api key") for _ in batch]

        return await asyncio.gather(*[self.translate_one(x) for x in batch])

    @staticmethod
    def get_name() -> str:
        return "DeepSeek"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key=lambda a: a[0].lower())
        options = list(map(lambda a: PluginSelectArgumentOption(a[0], a[1]), languages))

        return [
            PluginTextArgument(
                id="api_key",
                name="API Key",
                description="DeepSeek API key. Falls back to DEEPSEEK_API_KEY",
            ),
            PluginSelectArgument(
                id="target_lang",
                name="Target Language",
                description="The language to translate to",
                options=options,
                default="en",
            ),
            PluginSelectArgument(
                id="model",
                name="Model",
                description="The model to use",
                options=list(
                    map(
                        lambda a: PluginSelectArgumentOption(a[0], a[1]),
                        DeepSeekTranslator.MODELS,
                    )
                ),
                default=DeepSeekTranslator.MODELS[0][1],
            ),
            PluginTextArgument(
                id="temp",
                name="Temperature",
                description="Sampling temperature",
                default="1.3",
            ),
        ]
