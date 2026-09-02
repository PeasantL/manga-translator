import cv2
import torch
from numpy import ndarray
from translator.plugins.base import Cleaner
from translator.utils import in_paint_optimized, get_torch_device, hub_file


# LaMa, finetuned on anime and manga. The generic big-lama that used to be here
# was trained on photographs, which treats screentone as texture to reconstruct:
# erasing a line of dialogue off a toned panel left a smear where the letters
# had been, and a bubble edge crossing the mask came back bent. This checkpoint
# is torchscript, so it loads with torch alone and brings no inpainting library
# -- and no pillow ceiling -- along with it.
REPO = "TareHimself/AnimeMangaInpainting-torchscript"
WEIGHTS = "anime_manga_lama.pt"

# Shared so that a second LamaCleaner does not load 200 MB of weights again.
# Keyed by device because that is the only thing that differs between them.
_models = {}


def get_lama(device):
    key = str(device)

    if key not in _models:
        _models[key] = torch.jit.load(hub_file(REPO, WEIGHTS), map_location=device).eval()

    return _models[key]


def pad_to_modulo(tensor: torch.Tensor, modulo: int = 8) -> torch.Tensor:
    """Grow a tensor's last two dimensions to a multiple of modulo.

    LaMa downsamples three times, so a side that is not a multiple of 8 comes
    back the wrong size. Reflected rather than zero padded: a black margin at
    the edge of a window is something the model tries to paint away.
    """
    height, width = tensor.shape[-2:]

    return torch.nn.functional.pad(
        tensor,
        (0, (modulo - width % modulo) % modulo, 0, (modulo - height % modulo) % modulo),
        mode="reflect",
    )


class LamaCleaner(Cleaner):
    """Inpaints the text away with LaMa, on a checkpoint finetuned on manga"""

    def __init__(self, dilation="13") -> None:
        super().__init__()
        self.device = get_torch_device()
        self.lama = get_lama(self.device)
        self.dilation = int(dilation)

    @staticmethod
    def get_name() -> str:
        return "Lama Cleaner"

    def clean_with_lama(self, frame: ndarray, mask: ndarray) -> ndarray:
        """Paint over everything the mask marks, in one window of the page."""
        height, width = frame.shape[:2]

        image = (
            torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)[None]
            .float()
            .div(255)
        )

        # The mask arrives as a three channel image because it is built the same
        # shape as the page it came off; the model wants one channel of it.
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        holes = torch.from_numpy(mask)[None, None].float().div(255)

        with torch.inference_mode():
            painted = self.lama(
                pad_to_modulo(image).to(self.device),
                pad_to_modulo(holes).to(self.device),
            )

        # Back to the size that came in: whatever the padding added is not part
        # of the page.
        painted = painted[0, :, :height, :width].clamp(0, 1).mul(255).round()

        return cv2.cvtColor(
            painted.permute(1, 2, 0).to("cpu", torch.uint8).numpy(), cv2.COLOR_RGB2BGR
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
            inpaint_fun=self.clean_with_lama,
        )
