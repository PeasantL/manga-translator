import asyncio
import aiohttp
import traceback
import json
from translator.core.plugin import (
    PluginSelectArgument,
    PluginSelectArgumentOption,
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    PluginTextArgument,
)
from translator.utils import get_languages

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiTranslator(Translator):
    """Translates using Google's Gemini models, requires an api key"""

    MODELS = [
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ("Gemini 3.5 Flash", "gemini-3.5-flash"),
    ]

    DEFAULT_MODEL = MODELS[0][1]

    # Transient failures worth another attempt. Everything else (a bad key, a
    # malformed request, a safety block) will fail again exactly the same way,
    # so retrying it only delays the error.
    MAX_ATTEMPTS = 3
    RETRY_STATUSES = {429, 500, 502, 503, 504}

    def __init__(self, api_key="", target_lang="en", model=DEFAULT_MODEL) -> None:
        super().__init__()
        self.api_key = api_key
        self.target_lang = target_lang
        self.model = model

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key=lambda a: a[0].lower())
        options = list(map(lambda a: PluginSelectArgumentOption(a[0], a[1]), languages))

        return [
            PluginTextArgument(
                id="api_key", name="Api Key", description="Gemini Api Key"
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
                options=[
                    PluginSelectArgumentOption(name, value)
                    for name, value in GeminiTranslator.MODELS
                ],
                default=GeminiTranslator.DEFAULT_MODEL,
            ),
        ]

    def build_body(self, result: OcrResult) -> dict:
        message = f"{result.language.upper()} to {self.target_lang.upper()}\n{result.text}"

        return {
            "contents": [
                {"role": "user", "parts": [{"text": "EN to JA\nHello World"}]},
                {"role": "model", "parts": [{"text": "こんにちは世界"}]},
                {"role": "user", "parts": [{"text": message}]},
            ]
        }

    async def do_api(self, result: OcrResult):
        if self.api_key is None or len(self.api_key.strip()) == 0:
            return TranslatorResult("Need Gemini api key")

        if len(result.text.strip()) == 0:
            return TranslatorResult("")

        uri = f"{API_ROOT}/{self.model}:generateContent"
        body = json.dumps(self.build_body(result))

        try:
            async with aiohttp.ClientSession() as session:
                for attempt in range(GeminiTranslator.MAX_ATTEMPTS):
                    async with session.post(
                        uri,
                        headers={
                            "Content-Type": "application/json",
                            # In the header rather than the query string, so the
                            # key does not end up in proxy and access logs.
                            "x-goog-api-key": self.api_key,
                        },
                        data=body,
                    ) as response:
                        status = response.status
                        data = await response.json()

                    if "candidates" in data:
                        return TranslatorResult(
                            data["candidates"][0]["content"]["parts"][0]["text"],
                            lang_code=self.target_lang,
                        )

                    if "promptFeedback" in data:
                        print(
                            "Gemini refused to translate for safety reasons :",
                            data["promptFeedback"],
                        )
                        return TranslatorResult(
                            "Gemini failed to translate for safety reasons"
                        )

                    message = data.get("error", {}).get("message", "unknown error")

                    if status not in GeminiTranslator.RETRY_STATUSES:
                        print(f"Gemini error {status}: {message}")
                        return TranslatorResult("Failed To Get Translation")

                    is_last = attempt == GeminiTranslator.MAX_ATTEMPTS - 1
                    if is_last:
                        print(
                            f"Gemini error {status} after "
                            f"{GeminiTranslator.MAX_ATTEMPTS} attempts: {message}"
                        )
                        return TranslatorResult("Failed To Get Translation")

                    # Back off before trying again: 0.5s, then 1s.
                    await asyncio.sleep(0.5 * (2**attempt))
        except:
            traceback.print_exc()
            return TranslatorResult("Failed To Get Translation")

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.gather(*[self.do_api(x) for x in batch])

    @staticmethod
    def get_name() -> str:
        return "Gemini"
