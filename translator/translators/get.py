from translator.core.plugin import Translator
from translator.translators.deepl import DeepLTranslator
from translator.translators.deepseek import DeepSeekTranslator
from translator.translators.debug import DebugTranslator


def get_translators() -> list[Translator]:
    return list(
        filter(
            lambda a: a.is_valid(),
            [
                DeepLTranslator,
                DeepSeekTranslator,
                DebugTranslator,
            ],
        )
    )
