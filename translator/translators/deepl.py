import traceback
import asyncio
import aiohttp
from translator.core.plugin import (
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    PluginTextArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)
from translator.utils import get_languages

FREE_API = "https://api-free.deepl.com/v2/translate"
PRO_API = "https://api.deepl.com/v2/translate"


class DeepLTranslator(Translator):
    """The Best after GPT but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_token=None, target_lang="en") -> None:
        super().__init__()
        self.auth_token = auth_token
        self.target_lang = target_lang

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        languages = get_languages()
        languages.sort(key=lambda a: a[0].lower())
        options = list(map(lambda a: PluginSelectArgumentOption(a[0], a[1]), languages))

        return [
            PluginTextArgument(
                id="auth_token", name="Auth Token", description="DeepL Api Auth Token"
            ),
            PluginSelectArgument(
                id="target_lang",
                name="Target Language",
                description="The language to translate to",
                options=options,
                default="en",
            ),
        ]

    @property
    def endpoint(self) -> str:
        # DeepL issues free-tier keys with a ":fx" suffix, and they are only
        # valid against the free host.
        if self.auth_token.strip().endswith(":fx"):
            return FREE_API
        return PRO_API

    def deepl_target_lang(self) -> str:
        code = self.target_lang.strip().upper()
        # DeepL wants a regional variant for English targets.
        return "EN-US" if code == "EN" else code

    async def do_api(self, result: OcrResult):
        if self.auth_token is None or len(self.auth_token.strip()) == 0:
            return TranslatorResult("Need DeepL Auth")

        if len(result.text.strip()) == 0:
            return TranslatorResult("")

        # Sent as a form body rather than interpolated into the query string.
        # requote_uri leaves &, = and # alone, so any of those in the OCR output
        # used to corrupt or truncate the request.
        form = {
            "target_lang": self.deepl_target_lang(),
            "text": result.text,
        }

        # No source_lang means DeepL detects it, which is the right behaviour
        # when the OCR backend did not report one.
        if len(result.language.strip()) > 0:
            form["source_lang"] = result.language.strip().upper()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers={"Authorization": f"DeepL-Auth-Key {self.auth_token}"},
                    data=form,
                ) as response:
                    status = response.status
                    data = await response.json()

            if status != 200 or "translations" not in data:
                print(f"DeepL error {status}: {data.get('message', data)}")
                return TranslatorResult("Failed To Get Translation")

            return TranslatorResult(
                data["translations"][0]["text"], lang_code=self.target_lang
            )
        except:
            traceback.print_exc()
            return TranslatorResult("Failed To Get Translation")

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.gather(*[self.do_api(x) for x in batch])

    @staticmethod
    def get_name() -> str:
        return "DeepL"
