from numpy import ndarray
from translator.plugins.base import Cleaner, PluginArgument, PluginTextArgument
from translator.utils import in_paint_optimized, cv2_to_pil, pil_to_cv2


# Shared across instances so that a new LamaCleaner per web request does not
# reload 196 MB of weights. The import stays inside the accessor to keep merely
# importing this module from pulling in torch hub and downloading the checkpoint.
_lama = None


def get_lama():
    global _lama

    if _lama is None:
        from simple_lama_inpainting import SimpleLama

        _lama = SimpleLama()

    return _lama


class LamaCleaner(Cleaner):
    """Inpaints the text away with LaMa"""

    def __init__(self, dilation="13") -> None:
        super().__init__()
        self.lama = get_lama()
        self.dilation = int(dilation)

    @staticmethod
    def get_name() -> str:
        return "Lama Cleaner"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [PluginTextArgument(id="dilation", name="Mask Dilation",description="The dilation used for the text mask", default="13")]
    
    def clean_with_lama(self,frame,mask):
        return pil_to_cv2(
                self.lama(cv2_to_pil(frame), cv2_to_pil(mask).convert("L"))
            )
    
    async def clean(
        self,
        frame: ndarray,
        mask: ndarray,
        detection_results: list[tuple[tuple[int, int, int, int], str, float]] = [],
    ) -> tuple[ndarray, ndarray]:
        return in_paint_optimized(
            frame=frame,
            mask=mask,
            filtered=detection_results,
            mask_dilation_kernel_size=self.dilation,
            inpaint_fun=lambda f, m: self.clean_with_lama(f,m),
        )
