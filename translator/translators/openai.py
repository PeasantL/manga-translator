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


class OpenAiTranslator(Translator):
    """Uses an Open Ai Model for translation"""

    # Chat models that accept a temperature. The newer reasoning-style models
    # (the o-series, and the gpt-5 family over Chat Completions) reject any
    # temperature other than their default, and this backend sends one, so they
    # are deliberately not listed here.
    MODELS = [
        ("GPT 4o mini", "gpt-4o-mini"),
        ("GPT 4o", "gpt-4o"),
        ("GPT 4.1 mini", "gpt-4.1-mini"),
        ("GPT 4.1", "gpt-4.1"),
        ("GPT 4 Turbo", "gpt-4-turbo"),
    ]

    DEFAULT_MODEL = MODELS[0][1]
    DEFAULT_TEMPERATURE = "0.2"

    def __init__(
        self,
        api_key="",
        target_lang="en",
        model=DEFAULT_MODEL,
        temp=DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()
        import openai

        # Prefer the key the caller supplied (the UI collects one per backend,
        # the CLI takes one via --translator-args) and fall back to the
        # environment. This used to unconditionally overwrite the argument with
        # os.getenv, so a key passed in was discarded.
        self.api_key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")

        openai.api_key = self.api_key
        self.openai = openai
        self.target_lang = target_lang
        self.model = model
        self.temp = float(temp)

    async def translate_one(self, ocr_result: OcrResult):
        message = f"{ocr_result.language.upper()} to {self.target_lang.upper()}\n{ocr_result.text}"

        result = self.openai.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "user", "content": "EN to JA\nHello"},
                {"role": "assistant", "content": "こんにちは"},
                {"role": "user", "content": message},
            ],
        )
        return TranslatorResult(
            result.choices[0].message.content.strip(), self.target_lang
        )
    
    async def translate(self, batch: list[OcrResult]):
        if len(self.api_key.strip()) == 0:
            return [TranslatorResult("Need OpenAI api key") for _ in batch]

        return await asyncio.gather(*[self.translate_one(x) for x in batch])

    @staticmethod
    def get_name() -> str:
        return "Open AI"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key=lambda a: a[0].lower())
        options = list(map(lambda a: PluginSelectArgumentOption(a[0], a[1]), languages))

        return [
            PluginTextArgument(
                id="api_key", name="API Key", description="Your api Key"
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
                        OpenAiTranslator.MODELS,
                    )
                ),
                default=OpenAiTranslator.DEFAULT_MODEL,
            ),
            PluginTextArgument(
                id="temp",
                name="Temperature",
                description="Sampling temperature, 0 to 2. Lower is more literal.",
                default=OpenAiTranslator.DEFAULT_TEMPERATURE,
            ),
        ]
