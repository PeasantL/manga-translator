import time
import cv2
import numpy as np
from ultralytics import YOLO
from translator.utils import (
    mask_text_and_make_bubble_mask,
    get_bounds_for_text,
    TranslatorGlobals,
    has_white,
    get_model_path,
    require_model_file,
    apply_mask,
    reading_order_indices,
)
import traceback
import torch
import asyncio
from typing import Union
from translator.core.plugin import Drawable, Translator, Ocr, Drawer, Cleaner, TranslatorResult, OcrResult
from translator.cleaners.lama import LamaCleaner
from translator.drawers.horizontal import HorizontalDrawer


# server.py builds a FullConversion per request, which reloaded every model each
# time. The weights are read-only during inference, so one instance per path is
# enough for the whole process.
_yolo_models: dict[str, YOLO] = {}


def load_yolo(path: str) -> YOLO:
    resolved = require_model_file(path)

    if resolved not in _yolo_models:
        _yolo_models[resolved] = YOLO(resolved)

    return _yolo_models[resolved]


async def draw_page(
    frame: np.ndarray,
    draw_boxes: list[tuple[int, int, int, int]],
    translations: list[TranslatorResult],
    drawer: Drawer,
) -> np.ndarray:
    """Draw translations into an already cleaned page.

    Deliberately a plain function taking a drawer rather than a method on
    FullConversion: stage 6 needs no detection, cleaning or OCR model, so running
    it on its own should not have to load any of them.
    """
    if len(draw_boxes) == 0:
        return frame

    try:
        color = (
            TranslatorGlobals.COLOR_BLACK,
            TranslatorGlobals.COLOR_BLACK,
            False,
        )

        # The drawer says how much room the text needs. A translation too long
        # for its bubble is drawn over the surrounding art at a readable size,
        # on a backdrop, rather than being shrunk until nobody can read it.
        boxes = []
        to_draw = []

        for bbox, translation in zip(draw_boxes, translations):
            box, expanded = drawer.box_for(translation.text, bbox, frame.shape)

            (x1, y1, x2, y2) = box
            boxes.append(box)

            to_draw.append(
                Drawable(
                    color=color,
                    frame=frame[y1:y2, x1:x2].copy(),
                    translation=translation,
                    backdrop=expanded,
                )
            )

        drawn_frames = await drawer(to_draw)

        for bbox, drawn_frame in zip(boxes, drawn_frames):
            (x1, y1, x2, y2) = bbox
            drawn_frame, drawn_frame_mask = drawn_frame
            frame[y1:y2, x1:x2] = apply_mask(
                drawn_frame, frame[y1:y2, x1:x2], drawn_frame_mask
            )

        return frame
    except:
        traceback.print_exc()
        return frame


def align_translations(
    translations: list[TranslatorResult], expected: int
) -> list[TranslatorResult]:
    """Force a translator's output to the region count it was given.

    A translator returning the wrong number of results would otherwise shift every
    later page's dialogue onto the wrong bubbles.
    """
    if len(translations) == expected:
        return translations

    print(
        f"Translator returned {len(translations)} results for {expected} regions, "
        "padding the difference"
    )

    return (translations + [TranslatorResult("")] * expected)[:expected]


class PageLayout:
    """One page after cleaning, plus the text regions it contributes to the chapter.

    draw_boxes and text_crops are parallel and in reading order.
    """

    def __init__(
        self,
        frame: np.ndarray,
        draw_boxes: list[tuple[int, int, int, int]],
        text_crops: list[np.ndarray],
    ) -> None:
        self.frame = frame
        self.draw_boxes = draw_boxes
        self.text_crops = text_crops


class FullConversion:
    def __init__(
        self,
        detect_model: Union[str, None] = None,
        seg_model: Union[str, None] = None,
        translator: Union[Translator, None] = None,
        ocr: Union[Ocr, None] = None,
        drawer: Union[Drawer, None] = None,
        cleaner: Union[Cleaner, None] = None,
        translate_free_text: bool = False,
        device=None,
        yolo_device=None,
        debug=False,
    ) -> None:
        # Everything below is built here rather than in the signature. As default
        # arguments they were evaluated once at import time, so merely importing this
        # module downloaded and loaded the LaMa weights - even for `main.py --help`.
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if yolo_device is None:
            yolo_device = 0 if torch.cuda.is_available() else "cpu"

        if detect_model is None:
            detect_model = get_model_path("detection.pt")

        if seg_model is None:
            seg_model = get_model_path("segmentation.pt")

        self.device = device
        print("Pipeline created using",device)
        self.yolo_device = yolo_device
        self.segmentation_model = load_yolo(seg_model)
        self.detection_model = load_yolo(detect_model)

        self.translate_free_text = translate_free_text
        self.translator = translator if translator is not None else Translator()
        self.ocr = ocr if ocr is not None else Ocr()
        self.drawer = drawer if drawer is not None else HorizontalDrawer()
        self.debug = debug
        self.cleaner = cleaner if cleaner is not None else LamaCleaner()

    def filter_results(self, results, min_confidence=0.1):
        bounding_boxes = np.array(results.boxes.xyxy.cpu(), dtype="int")

        classes = np.array(results.boxes.cls.cpu(), dtype="int")

        confidence = np.array(results.boxes.conf.cpu(), dtype="float")

        raw_results: list[tuple[tuple[int, int, int, int], str, float]] = []

        for box, obj_class, conf in zip(bounding_boxes, classes, confidence):
            if conf >= min_confidence:
                raw_results.append((box, results.names[obj_class], conf))

        raw_results.sort(key=lambda a: 1 - a[2])

        return raw_results

    async def process_ml_results(self, detect_result, seg_result, frame):
        text_mask = np.zeros_like(frame, dtype=frame.dtype)

        if seg_result.masks is not None:  # Fill in segmentation results
            for seg in list(map(lambda a: a.astype("int"), seg_result.masks.xy)):
                cv2.fillPoly(text_mask, [seg], (255, 255, 255))

        detect_result = self.filter_results(detect_result)

        for bbox, cls, conf in detect_result:  # fill in text free results
            if cls == "text_free":
                (x1, y1, x2, y2) = bbox
                text_mask = cv2.rectangle(
                    text_mask, (x1, y1), (x2, y2), (255, 255, 255), -1
                )

        frame_clean, text_mask = await self.cleaner(
            frame=frame, mask=text_mask, detection_results=detect_result
        )  # segmentation_results.boxes.xyxy.cpu().numpy()

        return frame, frame_clean, text_mask, detect_result

    async def prepare_frame(self, detect_result, seg_result, input_frame) -> "PageLayout":
        """Stages 1 to 3 for one page: detect, clean, and collect its text regions.

        Returns the cleaned page plus the regions to translate, in reading order.
        OCR and translation deliberately do not happen here - they run once for the
        whole chapter so the translator can use the surrounding dialogue.
        """
        try:
            frame, frame_clean, text_mask, detect_result = await self.process_ml_results(
                detect_result, seg_result, input_frame
            )

            to_translate = []
            # First pass, mask all bubbles
            for bbox, cls, conf in detect_result:
                try:
                    color = (0, 0, 255) if cls == "text_free" else (0, 255, 0)

                    (x1, y1, x2, y2) = bbox

                    class_name = cls

                    bubble = frame[y1:y2, x1:x2]
                    bubble_clean = frame_clean[y1:y2, x1:x2]
                    bubble_text_mask = text_mask[y1:y2, x1:x2]

                    if class_name == "text_bubble":
                        if has_white(bubble_text_mask):
                            text_only, bubble_mask = mask_text_and_make_bubble_mask(
                                bubble, bubble_text_mask, bubble_clean
                            )

                            frame[y1:y2, x1:x2] = bubble_clean
                            text_draw_bounds = get_bounds_for_text(bubble_mask)

                            pt1, pt2 = text_draw_bounds

                            pt1_x, pt1_y = pt1
                            pt2_x, pt2_y = pt2

                            pt1_x += x1
                            pt2_x += x1
                            pt1_y += y1
                            pt2_y += y1

                            to_translate.append(
                                [(pt1_x, pt1_y, pt2_x, pt2_y), text_only]
                            )
                    else:
                        if self.translate_free_text:
                            free_text = frame[y1:y2, x1:x2]
                            if has_white(bubble_text_mask):
                                text_only, _ = mask_text_and_make_bubble_mask(
                                    free_text, bubble_text_mask, bubble_clean
                                )

                                to_translate.append([(x1, y1, x2, y2), text_only])

                            frame[y1:y2, x1:x2] = frame_clean[y1:y2, x1:x2]
                        else:
                            frame[y1:y2, x1:x2] = frame_clean[y1:y2, x1:x2]

                    if self.debug:
                        cv2.putText(
                            frame,
                            str(f"{cls} | {conf * 100:.1f}%"),
                            (x1, y1 - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            color,
                            2,
                        )
                except:
                    traceback.print_exc()

            # Detections arrive in confidence order, which is meaningless as reading
            # order. Sort them the way the page is read so that the translator sees
            # the dialogue in sequence.
            if len(to_translate) > 1:
                order = reading_order_indices([x[0] for x in to_translate])
                to_translate = [to_translate[i] for i in order]

            return PageLayout(
                frame=frame,
                draw_boxes=[x[0] for x in to_translate],
                text_crops=[x[1] for x in to_translate],
            )
        except:
            traceback.print_exc()
            return PageLayout(frame=input_frame, draw_boxes=[], text_crops=[])

    async def render_frame(self, page: "PageLayout", translations: list) -> np.ndarray:
        """Stage 6 for one page: draw the chapter's translations back into it."""
        return await draw_page(
            page.frame, page.draw_boxes, translations, self.drawer
        )

    async def clean_and_read(
        self,
        images: list[np.ndarray],
        detect_batch_size: int = 4,
        ocr_batch_size: int = 32,
        names: list[str] = None,
    ) -> tuple[list["PageLayout"], list[OcrResult]]:
        """Stages 1 to 4 for a chapter: detect, segment, clean, and read.

        Returns the cleaned pages with their regions in reading order, and the
        chapter's OCR results as one flat list in that same order. Nothing here
        touches the translator, so this half can run on its own.

        Detection and OCR are chunked so that a long chapter does not have to fit
        through those models all at once.
        """
        pages: list[PageLayout] = []

        total = len(images)
        labels = names if names is not None else [f"page {i + 1}" for i in range(total)]
        finished = 0
        start = time.time()

        async def prepare_one(detect_result, seg_result, frame, label):
            # Pages within a chunk are prepared concurrently, so the count is
            # incremented as each one lands rather than in index order.
            nonlocal finished

            page = await self.prepare_frame(
                detect_result=detect_result, seg_result=seg_result, input_frame=frame
            )

            finished += 1
            print(
                f"  [{finished}/{total}] cleaned {label}, "
                f"{len(page.draw_boxes)} regions"
            )

            return page

        for i in range(0, total, detect_batch_size):
            chunk = images[i:i + detect_batch_size]
            chunk_labels = labels[i:i + detect_batch_size]

            detections = self.detection_model(chunk, device=self.yolo_device, verbose=False)
            segmentations = self.segmentation_model(chunk, device=self.yolo_device, verbose=False)

            pages.extend(
                await asyncio.gather(
                    *[
                        prepare_one(detect_result, seg_result, frame, label)
                        for detect_result, seg_result, frame, label in zip(
                            detections, segmentations, chunk, chunk_labels
                        )
                    ]
                )
            )

        text_crops = [crop for page in pages for crop in page.text_crops]

        print(
            f"  cleaned {total} pages in {time.time() - start:.1f}s, "
            f"{len(text_crops)} text regions found"
        )

        ocr_results: list[OcrResult] = []

        if self.ocr and len(text_crops) > 0:
            start = time.time()

            for i in range(0, len(text_crops), ocr_batch_size):
                ocr_results.extend(await self.ocr(text_crops[i:i + ocr_batch_size]))
                print(f"  [{len(ocr_results)}/{len(text_crops)}] regions read")

            print(f"  read {len(text_crops)} regions in {time.time() - start:.1f}s")

        return pages, ocr_results

    async def translate_regions(self, ocr_results: list[OcrResult]) -> list[TranslatorResult]:
        """Stage 5 for a chapter: translate every region in one go.

        The whole chapter is passed to the translator as one ordered list, which
        is what lets it use a line's surrounding dialogue.
        """
        if not self.translator or len(ocr_results) == 0:
            return []

        start = time.time()
        translations = list(await self.translator(ocr_results))
        print(f"  translated {len(ocr_results)} regions in {time.time() - start:.1f}s")

        return align_translations(translations, len(ocr_results))

    async def render_pages(
        self, pages: list["PageLayout"], translations: list[TranslatorResult]
    ) -> list[np.ndarray]:
        """Stage 6 for a chapter: give each page its slice of the translations."""
        start = time.time()

        results = []
        offset = 0

        for page in pages:
            count = len(page.draw_boxes)

            if translations:
                results.append(
                    await self.render_frame(page, translations[offset:offset + count])
                )
            else:
                results.append(page.frame)

            offset += count

        print(f"  drew {len(results)} pages in {time.time() - start:.1f}s")

        return results

    async def __call__(
        self,
        images: list[np.ndarray],
        detect_batch_size: int = 4,
        ocr_batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Convert a chapter end to end, stages 1 to 6."""
        total_start = time.time()

        pages, ocr_results = await self.clean_and_read(
            images, detect_batch_size=detect_batch_size, ocr_batch_size=ocr_batch_size
        )
        translations = await self.translate_regions(ocr_results)
        results = await self.render_pages(pages, translations)

        print(f"  done in {time.time() - total_start:.1f}s")

        return results
