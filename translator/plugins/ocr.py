import numpy
import torch
from translator.utils import cv2_to_pil, get_torch_device
from translator.plugins.base import Ocr, OcrResult


# PaddleOCR-VL, finetuned on 100k text regions cut out of Manga109-s. The
# manga-ocr that used to be here reads Japanese and nothing else, and returns
# one unbroken line with its punctuation normalised to full width; this reads
# the scripts the base model knows, keeps the line breaks the letterer used,
# and gets ... and !! right instead of turning them into ．．．and ！！.
MODEL = "jzhang533/PaddleOCR-VL-For-Manga"

# What the model is asked for. The checkpoint is a vision language model with
# several tasks in it, and this is the one that reads text out of a crop.
PROMPT = "OCR:"

# Kana are Japanese and nothing else. Everything else in the CJK ideograph block
# is shared, so a line with no kana in it is read as Chinese.
KANA = ((0x3040, 0x30FF), (0x31F0, 0x31FF))
HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))

# Same reasoning as the cleaner: one loaded model per name, not one per request.
_models = {}


def dtype_for(device: torch.device) -> torch.dtype:
    """Half precision where it is actually faster, full where it is not.

    Pascal runs fp16 arithmetic at a fraction of its fp32 rate, so halving the
    weights on a card like that costs more time than the memory is worth. bf16,
    which is what the checkpoint ships as, it cannot do at all.
    """
    if device.type != "cuda":
        return torch.float32

    major, _ = torch.cuda.get_device_capability(device)

    return torch.float16 if major >= 7 else torch.float32


def get_model(name: str):
    if name not in _models:
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = get_torch_device()

        model = (
            AutoModelForCausalLM.from_pretrained(
                name,
                dtype=dtype_for(device),
                trust_remote_code=True,
                # The checkpoint asks for flash attention, which is not built
                # for every card this runs on. Torch's own kernel needs no wheel
                # and holds the attention of a page-sized crop in a fraction of
                # the memory the fallback does.
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )

        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        _models[name] = (model, processor, device)

    return _models[name]


def in_ranges(character: str, ranges) -> bool:
    return any(low <= ord(character) <= high for low, high in ranges)


def language_of(text: str) -> str:
    """Which of the languages this can read a line is in, by its script."""
    if any(in_ranges(c, KANA) for c in text):
        return "ja"

    if any(in_ranges(c, HAN) for c in text):
        return "zh"

    return ""


def join_lines(text: str) -> str:
    """One line out of the several a bubble was lettered in.

    The model returns the text as it is set, one line per line of lettering,
    which is a fact about the bubble rather than about the sentence. Japanese
    and Chinese are set without spaces, so their lines are run together; a line
    break between two latin words is a space.
    """
    joined = ""

    for line in (line.strip() for line in text.splitlines()):
        if len(line) == 0:
            continue

        if len(joined) == 0:
            joined = line
            continue

        cjk = in_ranges(joined[-1], KANA + HAN) and in_ranges(line[0], KANA + HAN)
        joined += line if cjk else f" {line}"

    return joined


class ComicOcr(Ocr):
    """Reads the text in a comic panel"""

    # Stated rather than implied so a caller can ask before sending a chapter
    # this cannot read. The finetuning was done on Japanese manga; Chinese is
    # what the base model brought with it, and is the weaker of the two.
    READS = frozenset({"ja", "zh"})

    def __init__(self, model=MODEL, max_new_tokens="256") -> None:
        super().__init__()
        self.model, self.processor, self.device = get_model(model)
        self.max_new_tokens = int(max_new_tokens)

    def read(self, crop) -> str:
        """Read one crop, as a PIL image.

        One at a time rather than in batches. Crops sent through together come
        back with each other's words in them -- the model packs the images of a
        batch into one sequence and does not keep them apart -- and on a chapter
        of bubbles a batch of four was no faster than four single reads anyway.
        """
        prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt], images=[crop], return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Everything up to the prompt's length is the prompt being echoed back.
        written = inputs["input_ids"].shape[1]

        return self.processor.decode(
            generated[0][written:], skip_special_tokens=True
        ).strip()

    async def do_ocr(self, batch: list[numpy.ndarray]):
        lines = [
            join_lines(self.read(cv2_to_pil(frame).convert("RGB"))) for frame in batch
        ]

        return [OcrResult(line, language_of(line)) for line in lines]

    @staticmethod
    def get_name() -> str:
        return "Comic Ocr"


class NoOcr(Ocr):
    """Skips OCR. Use it to clean a page without translating it"""

    def __init__(self) -> None:
        super().__init__()

    async def do_ocr(self, batch: list[numpy.ndarray]):
        return [OcrResult("", "") for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "No Ocr"
