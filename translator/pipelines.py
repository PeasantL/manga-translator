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
from translator.core.plugin import Drawable, Translator, Ocr, Drawer, Cleaner
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

        start = time.time()

        frame_clean, text_mask = await self.cleaner(
            frame=frame, mask=text_mask, detection_results=detect_result
        )  # segmentation_results.boxes.xyxy.cpu().numpy()

        print(f"Inpainting => {time.time() - start} seconds")

        return frame, frame_clean, text_mask, detect_result

    async def process_frame(self, detect_result, seg_result, input_frame):
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

            # third pass, draw text
            draw_colors = [
                (
                    TranslatorGlobals.COLOR_BLACK,
                    TranslatorGlobals.COLOR_BLACK,
                    False,
                )
                for _ in to_translate
            ]

            start = time.time()

            to_draw = []

            if self.translator and self.ocr and len(to_translate) > 0:
                bboxes,images = zip(*to_translate)

                #the ocr result, check here for the translation 
                ocr_results = await self.ocr(list(images))

                translation_results = await self.translator(ocr_results)

                to_draw = []
                for bbox,translation,color in zip(bboxes,translation_results,draw_colors):

                    (x1, y1, x2, y2) = bbox
                    draw_area = frame[y1:y2, x1:x2].copy()

                    to_draw.append(Drawable(color=color,frame=draw_area,translation=translation))

                    

                print(f"Ocr And Translation => {time.time() - start} seconds")

                start = time.time()

                drawn_frames = await self.drawer(to_draw)


                for bbox, drawn_frame in zip(bboxes,drawn_frames):
                    (x1, y1, x2, y2) = bbox
                    drawn_frame,drawn_frame_mask = drawn_frame
                    frame[y1:y2, x1:x2] = apply_mask(drawn_frame,frame[y1:y2, x1:x2],drawn_frame_mask)

                    

                print(f"Drawing => {time.time() - start} seconds")
            return frame
        except:
            traceback.print_exc()
            return input_frame

    async def __call__(
        self,
        images: list[np.ndarray],
    ) -> list[np.ndarray]:
        total_start = time.time()
        start = time.time()
        to_process = [
            x
            for x in zip(
                self.detection_model(images, device=self.yolo_device, verbose=False),
                self.segmentation_model(images, device=self.yolo_device, verbose=False),
                images,
            )
        ]

        print(f"Yolov8 Models => {time.time() - start} seconds")

        tasks = [self.process_frame(detect_result=detect_result,seg_result=seg_result,input_frame=frame) for detect_result, seg_result, frame in to_process]
        results = await asyncio.gather(*tasks)

        print(f"Total Process => {time.time() - total_start} seconds")
        return results