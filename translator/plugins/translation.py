import os
import re
import sys
from translator.plugins.base import Translator, TranslatorResult, OcrResult

# Matches "12. some text", "12) some text", "12: some text". The separator is
# required so that a translation which itself opens with a number is not mistaken
# for a new entry.
NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.):\]]\s*(.*)$")

# What every prompt says, whatever the source. The shape of the request and the
# shape of the reply do not depend on the language being read, and only the
# reply format keeps the numbering that puts each line back in its own bubble.
_COMMON_PROMPT = (
    "Use the whole list for context: who is speaking, the tone, running jokes, "
    "names, and sentences that continue from one bubble into the next. Keep names "
    "and terms consistent across the chapter.\n\n"
    "Make ONLY one pass over the lines while thinking. Read them in order, settle each "
    "line as you reach it, and move on. NEVER go back over lines you have already "
    "settled. DO NOT draft the whole translation twice - the reasoning shares "
    "a token budget with the answer, and a second pass spends what the answer "
    "needs. Think only once, then answer.\n\n"
    "Reply with exactly one line per input line, formatted as `<number>. "
    "<translation>`, in the same order and numbered the same way. Output nothing "
    "else: no blank lines, no commentary, no notes, no romanisation, no quotes "
    "around the translation, and no alternative renderings. Give a single "
    "translation per line. If a line cannot be translated, repeat it unchanged "
    "after its number."
)

# The opening of the prompt, which is the part that says what is being read.
# Separate prompts per source rather than one that names the language, because
# what a translator needs telling differs by more than the language does: the
# conventions that survive into English, and the ones that must not, are not
# the same for a Japanese manga as for a Chinese manhua.
_SOURCE_PROMPTS = {
    "ja": (
        "You translate Japanese manga into {target}. You are given the dialogue "
        "of one chapter as numbered lines, in reading order.\n\n"
        "Keep Japanese given-name order and romanise names in Hepburn. Keep "
        "honorifics (-san, -kun, -chan, -sama, senpai) attached to names rather "
        "than translating them into English forms of address, and keep the "
        "register they imply. Sound effects and interjections written in kana "
        "become English sound words, not romaji.\n\n"
    ),
    "zh": (
        "You translate Chinese manhua into {target}. You are given the dialogue "
        "of one chapter as numbered lines, in reading order. The text may be "
        "simplified or traditional; read either.\n\n"
        "Romanise names in pinyin without tone marks, surname first, and keep "
        "that order throughout. Do not invent Japanese honorifics -- render "
        "Chinese forms of address (\u54e5, \u59d0, \u524d\u8f88, \u5e08\u5085, and the rest) as the English "
        "a speaker would actually use, or as part of the name where that reads "
        "better, and keep the relative status they carry. Chengyu and other set "
        "phrases become the nearest natural English idiom rather than a literal "
        "gloss; leave no pinyin in the output.\n\n"
    ),
}

# What to read anything else as. Every source this OCRs is one of the above, so
# this is only reached when a caller names something unexpected -- better a
# working generic prompt than a KeyError in the middle of a chapter.
_GENERIC_PROMPT = (
    "You translate comics into {target}. You are given the dialogue of one "
    "chapter as numbered lines, in reading order.\n\n"
)


def system_prompt(source_lang: str, target_lang: str = "en") -> str:
    """The prompt for reading this language, plus the rules common to all of them."""
    opening = _SOURCE_PROMPTS.get((source_lang or "").lower(), _GENERIC_PROMPT)

    return opening.format(target=(target_lang or "en").upper()) + _COMMON_PROMPT


def console_safe(text: str) -> str:
    """Make model output printable on whatever encoding stdout happens to have.

    Windows consoles default to cp1252, which cannot encode Japanese. The
    reasoning stream quotes the source lines back, so printing it raw crashes the
    run on exactly the input this tool exists for.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


class DeepSeekTranslator(Translator):
    """Translates a whole chapter at once using the DeepSeek API"""

    # DeepSeek serves an OpenAI-compatible API, so the openai client works
    # against it once the base url is pointed here.
    BASE_URL = "https://api.deepseek.com"

    MODELS = [
        ("DeepSeek V4 Pro", "deepseek-v4-pro"),
        ("DeepSeek V4 Flash", "deepseek-v4-flash"),
    ]

    # Both v4 models reason, and both take reasoning_effort: high, medium or
    # low. Reasoning shares the token budget with the answer, so raising it may
    # need max_tokens raised with it.

    def __init__(
        self,
        api_key="",
        target_lang="en",
        source_lang="",
        model=MODELS[0][1],
        temp="1.3",
        max_lines="200",
        stream="true",
        reasoning_effort="low",
        max_tokens="16384",
    ) -> None:
        super().__init__()
        from openai import AsyncOpenAI

        # Passed as an argument by anything constructing this directly; falls
        # back to the environment so a .env file works for the CLI.
        key = (api_key or "").strip() or os.getenv("DEEPSEEK_API_KEY", "")

        self.client = (
            AsyncOpenAI(api_key=key, base_url=DeepSeekTranslator.BASE_URL)
            if key
            else None
        )
        self.target_lang = target_lang
        # What the pages are in. Left empty to go by what the OCR reported,
        # which is all the CLI has to go on; a caller that knows better -- a
        # library that has the book's own LanguageISO -- sets it and wins,
        # because the OCR only ever reports the one language it can read.
        self.source_lang = (source_lang or "").strip().lower()
        self.model = model
        self.temp = float(temp)
        self.max_lines = max(1, int(max_lines))
        self.stream = str(stream).strip().lower() not in ("false", "0", "no", "off")
        self.reasoning_effort = str(reasoning_effort).strip().lower()
        self.max_tokens = int(max_tokens)

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
        source_lang = self.source_lang or chunk[0][1].language or "ja"

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

        reply = await self.ask(request, source_lang)
        by_position = self.parse_numbered(reply)

        translated = []
        missing = []

        for position in range(1, len(chunk) + 1):
            text = by_position.get(position, "").strip()
            translated.append(text)

            if len(text) == 0:
                missing.append(position)

        # A line the model dropped or mangled is left empty and its bubble is
        # left as it was cleaned. Asking again for the same line rarely produces
        # a different answer, and doing it one line at a time costs a request
        # each and throws away the chapter context that makes the answer good.
        if len(missing) > 0:
            print(
                f"DeepSeek did not return {len(missing)} of {len(chunk)} lines: "
                f"{', '.join(str(p) for p in missing)}. Those bubbles are left "
                "empty - fill them in in translated.json and re-run -s draw"
            )

        return translated

    def request_arguments(self, messages: list[dict]) -> dict:
        """The keywords both the streaming and non-streaming calls send."""
        return {
            "model": self.model,
            "temperature": self.temp,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
        }

    async def ask(self, content: str, source_lang: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt(source_lang or self.source_lang, self.target_lang),
            },
            {"role": "user", "content": content},
        ]

        if not self.stream:
            result = await self.client.chat.completions.create(
                **self.request_arguments(messages)
            )

            return result.choices[0].message.content or ""

        return await self.ask_streaming(messages)

    async def ask_streaming(self, messages: list[dict]) -> str:
        """Stream the reply, showing the reasoning and the lines as they arrive.

        The v4 models return their chain of thought in reasoning_content, which is
        shown as it arrives; the translated lines follow. reasoning_content is
        read with getattr so a model or endpoint that does not send it is not an
        error, it just shows nothing under "thinking".
        """
        stream = await self.client.chat.completions.create(
            **self.request_arguments(messages), stream=True
        )

        answer = []
        pending = ""
        thinking = False
        finish_reason = None

        async for chunk in stream:
            if len(chunk.choices) == 0:
                continue

            finish_reason = chunk.choices[0].finish_reason or finish_reason
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning:
                if not thinking:
                    print("  thinking: ", end="", flush=True)
                    thinking = True

                print(console_safe(reasoning), end="", flush=True)

            if delta.content:
                if thinking:
                    print(flush=True)
                    thinking = False

                answer.append(delta.content)
                pending += delta.content

                # Emit whole lines only, so a translation is never shown split
                # across two prints.
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)

                    if len(line.strip()) > 0:
                        print(f"  {console_safe(line.strip())}")

        if thinking:
            print(flush=True)

        if len(pending.strip()) > 0:
            print(f"  {console_safe(pending.strip())}")

        if finish_reason == "length":
            # Reasoning tokens count towards max_tokens, so a high reasoning
            # effort can eat the whole budget before any translation is written.
            print(
                f"  reply hit the {self.max_tokens} token cap and was cut off. "
                "Raise max_tokens, or lower reasoning_effort or max_lines"
            )

        return "".join(answer)

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

class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, text="") -> None:
        super().__init__()
        self.to_write = text

    async def translate(self, batch: list[OcrResult]):
        return [TranslatorResult(self.to_write) for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Custom Text"
