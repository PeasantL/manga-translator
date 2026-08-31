import os
import re
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

# Matches "12. some text", "12) some text", "12: some text". The separator is
# required so that a translation which itself opens with a number is not mistaken
# for a new entry.
NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.):\]]\s*(.*)$")

SYSTEM_PROMPT = (
    "You translate manga. You are given the dialogue of one chapter as numbered "
    "lines, in reading order.\n\n"
    "Use the whole list for context: who is speaking, the tone, running jokes, "
    "names, and sentences that continue from one bubble into the next. Keep names "
    "and terms consistent across the chapter.\n\n"
    "Reply with exactly one line per input line, formatted as `<number>. "
    "<translation>`, in the same order and numbered the same way. Output nothing "
    "else: no blank lines, no commentary, no notes, no romanisation, no quotes "
    "around the translation, and no alternative renderings. Give a single "
    "translation per line. If a line cannot be translated, repeat it unchanged "
    "after its number."
)


class DeepSeekTranslator(Translator):
    """Translates a whole chapter at once using the DeepSeek API"""

    # DeepSeek serves an OpenAI-compatible API, so the openai client works
    # against it once the base url is pointed here.
    BASE_URL = "https://api.deepseek.com"

    MODELS = [
        ("DeepSeek Chat", "deepseek-chat"),
        ("DeepSeek Reasoner", "deepseek-reasoner"),
    ]

    def __init__(
        self,
        api_key="",
        target_lang="en",
        model=MODELS[0][1],
        temp="1.3",
        max_lines="200",
    ) -> None:
        super().__init__()
        from openai import AsyncOpenAI

        # The UI sends the key as an argument; fall back to the environment so
        # a .env file works for the CLI.
        key = (api_key or "").strip() or os.getenv("DEEPSEEK_API_KEY", "")

        self.client = (
            AsyncOpenAI(api_key=key, base_url=DeepSeekTranslator.BASE_URL)
            if key
            else None
        )
        self.target_lang = target_lang
        self.model = model
        self.temp = float(temp)
        self.max_lines = max(1, int(max_lines))

    async def translate(self, batch: list[OcrResult]):
        if self.client is None:
            return [TranslatorResult("Need DeepSeek api key") for _ in batch]

        results = [TranslatorResult("", self.target_lang) for _ in batch]

        # Blank regions still occupy a slot, so that the caller's results line up
        # with its bubbles, but there is nothing to send for them.
        pending = [(i, r) for i, r in enumerate(batch) if len(r.text.strip()) > 0]

        if len(pending) == 0:
            return results

        # Chapters longer than max_lines are split, and each request after the
        # first is given the previous lines so the context is not lost at the seam.
        previous = []

        for start in range(0, len(pending), self.max_lines):
            chunk = pending[start:start + self.max_lines]

            translated = await self.translate_chunk(chunk, previous)

            for (index, _), text in zip(chunk, translated):
                results[index] = TranslatorResult(text, self.target_lang)

            previous = [
                (source.text, results[index].text) for index, source in chunk
            ][-10:]

        return results

    async def translate_chunk(
        self, chunk: list[tuple[int, OcrResult]], previous: list[tuple[str, str]]
    ) -> list[str]:
        source_lang = chunk[0][1].language or "ja"

        request = (
            f"Translate from {source_lang.upper()} to {self.target_lang.upper()}.\n\n"
        )

        if len(previous) > 0:
            already = "\n".join(f"{a} => {b}" for a, b in previous)
            request += (
                "Earlier lines of this chapter, already translated. They are context "
                f"only, do not translate or repeat them:\n{already}\n\n"
            )

        numbered = "\n".join(
            f"{position}. {source.text}"
            for position, (_, source) in enumerate(chunk, start=1)
        )
        request += f"Translate these {len(chunk)} lines:\n{numbered}"

        reply = await self.ask(request)
        by_position = self.parse_numbered(reply)

        translated = []
        missing = []

        for position in range(1, len(chunk) + 1):
            text = by_position.get(position, "").strip()
            translated.append(text)

            if len(text) == 0:
                missing.append(position)

        # A dropped or malformed line would leave a bubble empty, so fall back to
        # translating those on their own.
        if len(missing) > 0:
            print(
                f"DeepSeek did not return {len(missing)} of {len(chunk)} lines, "
                "retrying those individually"
            )

            for position in missing:
                source = chunk[position - 1][1]
                single = await self.ask(
                    f"Translate from {source_lang.upper()} to "
                    f"{self.target_lang.upper()}. Reply with the translation only.\n"
                    f"{source.text}"
                )
                translated[position - 1] = single.strip()

        return translated

    async def ask(self, content: str) -> str:
        result = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        return result.choices[0].message.content or ""

    @staticmethod
    def parse_numbered(reply: str) -> dict[int, str]:
        """Pull `<number>. <translation>` lines out of a reply.

        A line that does not start with a number is treated as a continuation of
        the previous one, which is how a translation containing a line break
        arrives.
        """
        by_position: dict[int, str] = {}
        current = None

        for line in reply.splitlines():
            match = NUMBERED_LINE.match(line)

            if match is not None:
                current = int(match.group(1))
                by_position[current] = match.group(2).strip()
            elif current is not None and len(line.strip()) > 0:
                by_position[current] = f"{by_position[current]} {line.strip()}".strip()

        return by_position

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
            PluginTextArgument(
                id="max_lines",
                name="Lines Per Request",
                description="How many lines to send in one request. A chapter with "
                "more than this is split, with the previous lines carried over as "
                "context",
                default="200",
            ),
        ]
