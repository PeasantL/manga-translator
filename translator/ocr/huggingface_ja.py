import numpy
import torch
import re
import jaconv
from transformers import pipeline
from translator.utils import cv2_to_pil, get_torch_device
from translator.core.plugin import Ocr, OcrResult


class JapaneseOcr(Ocr):
    """Only Supports Japanese"""

    def __init__(self,model='TareHimself/manga-ocr-base') -> None:
        super().__init__()
        self.pipeline = pipeline("image-to-text", model=model, device=get_torch_device())
    
    async def do_ocr(self, batch: list[numpy.ndarray]):

        with torch.inference_mode():
            frames = [cv2_to_pil(x).convert('L').convert('RGB') for x in batch]
            results = self.pipeline(frames,max_new_tokens=300)

            return [OcrResult(self._post_process(x[0]['generated_text']), "ja") for x in results]
    
    def _post_process(self,text: str):
        text = ''.join(text.split())
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        text = jaconv.h2z(text, ascii=True, digit=True)

        return text

    @staticmethod
    def get_name() -> str:
        return "Japanese Ocr"
