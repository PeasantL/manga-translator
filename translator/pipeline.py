import time
import cv2
import numpy as np
from ultralytics import YOLO
from translator.utils import (
    make_bubble_mask,
    get_bounds_for_text,
    has_white,
    hub_file,
    cv2_to_pil,
    apply_mask,
    reading_order_indices,
    measure_region_colors,
    drawing_colors,
)
import traceback
import torch
import asyncio
from typing import Callable, Union
from translator.plugins import (
    Drawable,
    Translator,
    Ocr,
    Drawer,
    Cleaner,
    TranslatorResult,
    OcrResult,
    LamaCleaner,
    HorizontalDrawer,
)


# The detector. RT-DETR-v2, trained on 11k manga, webtoon, manhua and western
# comic pages, and it reports three things where the YOLOv8 model that used to
# be here reported two: the balloon, the text inside the balloon, and text that
# is in no balloon at all. Having the text apart from the balloon is what lets
# the reader be given the lettering and the letterer the whole balloon.
DETECT_REPO = "ogkalu/comic-text-and-bubble-detector"

# The segmenter. YOLOv8m trained at 1024 pixels on the same sorts of page; it
# marks text pixels wherever they are, and what it marks is what gets erased.
SEGMENT_REPO = "ogkalu/comic-text-segmenter-yolov8m"
SEGMENT_WEIGHTS = "comic-text-segmenter.pt"

# How much of a free text box the segmenter has to have marked before the whole
# box is erased rather than just what it marked. See process_ml_results.
FREE_TEXT_COVERAGE = 0.2

# The weights are read-only during inference, so one instance per model is enough
# for the whole process, however many FullConversions are built over its life.
_yolo_models: dict[str, YOLO] = {}
_detectors: dict[str, tuple] = {}


def load_detector(repo: str, device):
    """The detection model and the processor that feeds it, loaded once."""
    key = f"{repo} on {device}"

    if key not in _detectors:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        _detectors[key] = (
            AutoModelForObjectDetection.from_pretrained(repo).to(device).eval(),
            AutoImageProcessor.from_pretrained(repo),
        )

    return _detectors[key]


def load_yolo(repo: str, filename: str = SEGMENT_WEIGHTS) -> YOLO:
    path = hub_file(repo, filename)

    if path not in _yolo_models:
        _yolo_models[path] = YOLO(path)

    return _yolo_models[path]


def clamp_box(box, width: int, height: int):
    """A box cut to the page, or None if there is nothing left of it.

    The detector answers in the coordinates of the page it was given, but it
    predicts them rather than reading them off, so a box around something at the
    edge can start a few pixels outside it. Negative coordinates index from the
    far end of an array, which turns a box at the left margin into a slice of
    the right one.
    """
    x1, y1, x2, y2 = box

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def middle_of(box) -> tuple[float, float]:
    (x1, y1, x2, y2) = box

    return (x1 + x2) / 2, (y1 + y2) / 2


def holds(box, point) -> bool:
    """Whether a box has a point inside it."""
    (x1, y1, x2, y2) = box
    x, y = point

    return x1 <= x <= x2 and y1 <= y <= y2


def crop_with_margin(frame: np.ndarray, box, margin: int = 4) -> np.ndarray:
    """A copy of what is inside a box, with a little of the page around it.

    A copy because the page is painted over region by region as it is cleaned,
    and whatever reads this has to see the lettering that was there. The margin
    is because a box drawn tight against the glyphs clips the strokes that lean
    out of them, and half a stroke reads as a different character.
    """
    height, width = frame.shape[:2]
    (x1, y1, x2, y2) = box

    return frame[
        max(0, y1 - margin):min(height, y2 + margin),
        max(0, x1 - margin):min(width, x2 + margin),
    ].copy()


async def draw_page(
    frame: np.ndarray,
    draw_boxes: list[tuple[int, int, int, int]],
    translations: list[TranslatorResult],
    drawer: Drawer,
    colors: list = None,
) -> np.ndarray:
    """Draw translations into an already cleaned page.

    Deliberately a plain function taking a drawer rather than a method on
    FullConversion: stage 6 needs no detection, cleaning or OCR model, so running
    it on its own should not have to load any of them.
    """
    if len(draw_boxes) == 0:
        return frame

    try:
        # A region whose colours were never measured is lettered black on white,
        # which is what every bubble was assumed to be before they were.
        if colors is None:
            colors = [(None, None) for _ in draw_boxes]

        # The drawer says how much room the text needs. A translation too long
        # for its bubble is drawn over the surrounding art at a readable size,
        # on a backdrop, rather than being shrunk until nobody can read it.
        boxes = []
        to_draw = []

        for bbox, translation, measured in zip(draw_boxes, translations, colors):
            box, expanded = drawer.box_for(translation.text, bbox, frame.shape)

            (x1, y1, x2, y2) = box
            boxes.append(box)

            to_draw.append(
                Drawable(
                    color=drawing_colors(*measured),
                    frame=frame[y1:y2, x1:x2].copy(),
                    translation=translation,
                    backdrop=expanded,
                    page_shape=frame.shape,
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

    draw_boxes, text_crops and colors are parallel and in reading order. Each
    entry in colors is the (text, background) pair measured off the page before
    the text was erased.
    """

    def __init__(
        self,
        frame: np.ndarray,
        draw_boxes: list[tuple[int, int, int, int]],
        text_crops: list[np.ndarray],
        colors: list = None,
    ) -> None:
        self.frame = frame
        self.draw_boxes = draw_boxes
        self.text_crops = text_crops
        self.colors = colors if colors is not None else [(None, None) for _ in draw_boxes]


class FullConversion:
    def __init__(
        self,
        detect_model: str = DETECT_REPO,
        seg_model: str = SEGMENT_REPO,
        translator: Union[Translator, None] = None,
        ocr: Union[Ocr, None] = None,
        drawer: Union[Drawer, None] = None,
        cleaner: Union[Cleaner, None] = None,
        translate_free_text: bool = False,
        min_confidence: float = 0.4,
        device=None,
        yolo_device=None,
        debug=False,
        progress: Union[Callable[[str, int, int], None], None] = None,
    ) -> None:
        # Everything below is built here rather than in the signature. As default
        # arguments they were evaluated once at import time, so merely importing this
        # module downloaded and loaded the LaMa weights - even for `main.py --help`.
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if yolo_device is None:
            yolo_device = 0 if torch.cuda.is_available() else "cpu"

        self.device = device
        print("Pipeline created using",device)
        self.yolo_device = yolo_device
        self.segmentation_model = load_yolo(seg_model)
        self.detection_model, self.detect_processor = load_detector(detect_model, device)

        self.translate_free_text = translate_free_text
        # Below this a detection is not reported at all. The detector answers
        # with a fixed number of guesses however little is on the page, so the
        # tail of that list is nothing at all seen faintly.
        self.min_confidence = min_confidence
        self.translator = translator if translator is not None else Translator()
        self.ocr = ocr if ocr is not None else Ocr()
        self.drawer = drawer if drawer is not None else HorizontalDrawer()
        self.debug = debug
        self.cleaner = cleaner if cleaner is not None else LamaCleaner()
        # Called as (stage, done, total) as the chapter goes through. The CLI
        # leaves it unset and reads the printed lines instead; a caller driving
        # this from a request has nowhere to read those, and a chapter takes
        # long enough that something has to be able to say where it has got to.
        self.progress = progress

    def _report(self, stage: str, done: int, total: int) -> None:
        """Tell the caller where the chapter has got to, if it asked."""
        if self.progress is not None:
            self.progress(stage, done, total)

    def detect(self, frames: list[np.ndarray]) -> list[list[tuple]]:
        """Stage 1 for a chunk of pages: what is on each of them, and where.

        One entry per page, each a list of ((x1, y1, x2, y2), class, confidence)
        sorted from the most confident down, which is the shape the rest of this
        file has always taken a page's detections in.
        """
        images = [cv2_to_pil(frame) for frame in frames]
        sizes = torch.tensor([image.size[::-1] for image in images]).to(self.device)

        with torch.inference_mode():
            inputs = self.detect_processor(images=images, return_tensors="pt").to(
                self.device
            )
            outputs = self.detection_model(**inputs)

        pages = self.detect_processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=self.min_confidence
        )

        names = self.detection_model.config.id2label
        detections = []

        for page, frame in zip(pages, frames):
            height, width = frame.shape[:2]

            found = [
                (
                    clamp_box([int(round(float(v))) for v in box], width, height),
                    names[int(label)],
                    float(score),
                )
                for box, label, score in zip(
                    page["boxes"].cpu(), page["labels"].cpu(), page["scores"].cpu()
                )
            ]

            found.sort(key=lambda a: 1 - a[2])
            detections.append([d for d in found if d[0] is not None])

        return detections

    def text_in(self, bubble, detections):
        """The box of the text a balloon holds, if the detector found any in it.

        A balloon with no text box inside it is one nobody spoke in -- an empty
        one, or a tail caught on its own -- and there is nothing there to read
        or to letter.
        """
        best = None

        for box, cls, conf in detections:
            if cls != "text_bubble" or not holds(bubble, middle_of(box)):
                continue

            if best is None or conf > best[1]:
                best = (box, conf)

        return None if best is None else best[0]

    async def process_ml_results(self, detections, seg_result, frame):
        text_mask = np.zeros_like(frame, dtype=frame.dtype)

        if seg_result.masks is not None:  # Fill in segmentation results
            for seg in list(map(lambda a: a.astype("int"), seg_result.masks.xy)):
                cv2.fillPoly(text_mask, [seg], (255, 255, 255))

        # Text outside a balloon used to be filled into the mask as a solid
        # rectangle and painted out whole. On this detector that erases the
        # chapter title and the artwork it is set over, because it finds titles
        # and logos where the old one saw nothing.
        #
        # So the two models have to agree. Where the segmenter marks a good part
        # of a free text box, it is a block of lettering and the whole box goes:
        # a mask that misses one stroke leaves half a character standing in the
        # artwork, which reads worse than a clean patch. Where it marks none of
        # it, the box is a logo or a piece of the drawing, and nothing goes.
        # There is no middle to speak of -- on the sample pages it is 60% of the
        # box or none of it -- so where the line falls between them hardly
        # matters.
        for bbox, cls, conf in detections:
            if cls != "text_free":
                continue

            (x1, y1, x2, y2) = bbox
            marked = text_mask[y1:y2, x1:x2]

            if marked.size > 0 and (marked > 0).mean() >= FREE_TEXT_COVERAGE:
                cv2.rectangle(text_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # The cleaner works a window at a time, and these are the windows: where
        # the detector found lettering. A balloon is not one of them, because the
        # text box inside it already is and the rest of a balloon is the empty
        # white around the words.
        windows = [d for d in detections if d[1] in ("text_bubble", "text_free")]

        frame_clean, text_mask = await self.cleaner(
            frame=frame, mask=text_mask, detection_results=windows
        )

        return frame, frame_clean, text_mask, detections

    async def prepare_frame(self, detections, seg_result, input_frame) -> "PageLayout":
        """Stages 1 to 3 for one page: detect, clean, and collect its text regions.

        Returns the cleaned page plus the regions to translate, in reading order.
        OCR and translation deliberately do not happen here - they run once for the
        whole chapter so the translator can use the surrounding dialogue.
        """
        try:
            frame, frame_clean, text_mask, detections = await self.process_ml_results(
                detections, seg_result, input_frame
            )

            to_translate = []
            # Which text boxes a balloon has already spoken for. Two balloons
            # drawn around the same words -- a bubble inside a bubble, or the
            # same one found twice -- would otherwise have that line translated
            # and lettered once each, and the most confident of them is the one
            # worth keeping. What is unclaimed at the end is text the detector
            # put in no balloon it also found, and that is read on its own
            # rather than dropped.
            claimed = set()

            for bbox, cls, conf in detections:
                try:
                    if cls == "bubble":
                        region, text_box = self.read_bubble(
                            bbox, detections, frame, frame_clean, text_mask
                        )

                        if region is not None and text_box not in claimed:
                            claimed.add(text_box)
                            to_translate.append(region)
                    elif cls == "text_free" and self.translate_free_text:
                        region = self.read_loose_text(
                            bbox, frame, frame_clean, text_mask
                        )

                        if region is not None:
                            to_translate.append(region)

                    if self.debug:
                        (x1, y1, x2, y2) = bbox
                        cv2.putText(
                            frame,
                            str(f"{cls} | {conf * 100:.1f}%"),
                            (x1, y1 - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 0, 255) if cls == "text_free" else (0, 255, 0),
                            2,
                        )
                except:
                    traceback.print_exc()

            # A balloon the detector missed while finding the text inside it
            # would otherwise take a line of dialogue out of the chapter. There
            # is no balloon to fit the translation to, so it is lettered into
            # the box the original text filled.
            for bbox, cls, conf in detections:
                if cls != "text_bubble" or bbox in claimed:
                    continue

                region = self.read_loose_text(bbox, frame, frame_clean, text_mask)

                if region is not None:
                    to_translate.append(region)

            # Every region is painted clean, whether or not it is being
            # translated, and only once everything above has read what it needs
            # off the page. Erasing the text is the one thing that happens
            # everywhere something was found.
            for bbox, cls, conf in detections:
                (x1, y1, x2, y2) = bbox
                frame[y1:y2, x1:x2] = frame_clean[y1:y2, x1:x2]

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
                colors=[x[2] for x in to_translate],
            )
        except:
            traceback.print_exc()
            return PageLayout(frame=input_frame, draw_boxes=[], text_crops=[])

    def read_bubble(self, bbox, detections, frame, frame_clean, text_mask):
        """One balloon: where to letter it, what it says, and in what colours.

        The box handed back is the balloon's interior rather than the box the old
        lettering filled, because a translation is not the shape of what it
        replaces and a balloon is the room there is for it. The crop handed back
        is the lettering itself, off the page as it was before cleaning.
        """
        (x1, y1, x2, y2) = bbox

        bubble = frame[y1:y2, x1:x2]
        bubble_clean = frame_clean[y1:y2, x1:x2]
        bubble_text_mask = text_mask[y1:y2, x1:x2]

        text_box = self.text_in(bbox, detections)

        if text_box is None or not has_white(bubble_text_mask):
            return None, None

        # Measured before the original text is painted over, because it is the
        # only moment the page still has the lettering on it.
        colors = measure_region_colors(bubble, bubble_clean, bubble_text_mask)
        crop = crop_with_margin(frame, text_box)

        (pt1_x, pt1_y), (pt2_x, pt2_y) = get_bounds_for_text(
            make_bubble_mask(bubble_clean)
        )

        room = (pt1_x + x1, pt1_y + y1, pt2_x + x1, pt2_y + y1)

        # Two balloons drawn against each other share a crop: the room inside
        # this one is measured on a picture with part of its neighbour in it,
        # and the widest clear rectangle in that picture can be the neighbour's.
        # A pair of small bubbles side by side came out lettered one on top of
        # the other that way, with the second left empty.
        #
        # The words that were read say which balloon is which. Room that does
        # not hold them is not this balloon's room, and the box the original
        # lettering filled is the answer to fall back on: it is, by definition,
        # somewhere a line of text fits.
        if not holds(room, middle_of(text_box)):
            room = text_box

        return [room, crop, colors], text_box

    def read_loose_text(self, bbox, frame, frame_clean, text_mask):
        """Text with no balloon around it, lettered back into its own box."""
        (x1, y1, x2, y2) = bbox

        section = frame[y1:y2, x1:x2]
        section_clean = frame_clean[y1:y2, x1:x2]
        section_mask = text_mask[y1:y2, x1:x2]

        if not has_white(section_mask):
            return None

        colors = measure_region_colors(section, section_clean, section_mask)

        return [(x1, y1, x2, y2), crop_with_margin(frame, bbox), colors]

    async def render_frame(self, page: "PageLayout", translations: list) -> np.ndarray:
        """Stage 6 for one page: draw the chapter's translations back into it."""
        return await draw_page(
            page.frame, page.draw_boxes, translations, self.drawer, page.colors
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

        async def prepare_one(detections, seg_result, frame, label):
            # Pages within a chunk are prepared concurrently, so the count is
            # incremented as each one lands rather than in index order.
            nonlocal finished

            page = await self.prepare_frame(
                detections=detections, seg_result=seg_result, input_frame=frame
            )

            finished += 1
            print(
                f"  [{finished}/{total}] cleaned {label}, "
                f"{len(page.draw_boxes)} regions"
            )
            self._report("clean", finished, total)

            return page

        for i in range(0, total, detect_batch_size):
            chunk = images[i:i + detect_batch_size]
            chunk_labels = labels[i:i + detect_batch_size]

            detections = self.detect(chunk)
            segmentations = self.segmentation_model(chunk, device=self.yolo_device, verbose=False)

            pages.extend(
                await asyncio.gather(
                    *[
                        prepare_one(detections, seg_result, frame, label)
                        for detections, seg_result, frame, label in zip(
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
                self._report("read", len(ocr_results), len(text_crops))

            print(f"  read {len(text_crops)} regions in {time.time() - start:.1f}s")

        return pages, ocr_results

    async def translate_regions(self, ocr_results: list[OcrResult]) -> list[TranslatorResult]:
        """Stage 5 for a chapter: translate every region in one go.

        The whole chapter is passed to the translator as one ordered list, which
        is what lets it use a line's surrounding dialogue.
        """
        if not self.translator or len(ocr_results) == 0:
            return []

        # Reported before rather than after: this is one request covering the
        # whole chapter, so there is no progress to be had within it, and a
        # caller showing "translating" wants it up while the wait is happening.
        self._report("translate", 0, len(ocr_results))

        start = time.time()
        translations = list(await self.translator(ocr_results))
        print(f"  translated {len(ocr_results)} regions in {time.time() - start:.1f}s")
        self._report("translate", len(ocr_results), len(ocr_results))

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
            self._report("draw", len(results), len(pages))

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
