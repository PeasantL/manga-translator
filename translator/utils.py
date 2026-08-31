import cv2
import os
import re
import math
import torch
import threading
import pycountry
import numpy as np
import asyncio
import inspect
import largestinteriorrectangle as lir
from typing import Union, Callable
from PIL import Image, ImageFont
from hyphen import Hyphenator
from collections import deque
import traceback


class TranslatorGlobals:
    COLOR_BLACK = np.array((0, 0, 0))
    COLOR_WHITE = np.array((255, 255, 255))

async def run_in_thread(func,*args,**kwargs):
    loop = asyncio.get_event_loop()
    task = asyncio.Future()
    def run():
        nonlocal loop
        nonlocal func
        nonlocal task
        
        try:
            result = func(*args,**kwargs)

            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except BaseException as e:
            # Without this the future is never resolved and the caller - a tornado
            # request handler - waits on it forever instead of returning a 500.
            loop.call_soon_threadsafe(task.set_exception,e)
            return

        loop.call_soon_threadsafe(task.set_result,result)
    
    task_thread = threading.Thread(group=None,daemon=True,target=run)
    task_thread.start()
    return await task

def run_in_thread_decorator(func):
    async def wrapper(*args,**kwargs):
        return await run_in_thread(func,*args,**kwargs)
    return wrapper


    
def get_torch_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

def simplify_lang_code(code: str) -> Union[str, None]:
    try:
        lang = pycountry.languages.lookup(code)

        return getattr(lang, "alpha_2", getattr(lang, "alpha_3", None))
    except:
        return code


def get_languages() -> list[tuple[str, str]]:
    return list(
        filter(
            lambda a: a[1] is not None,
            list(
                map(
                    lambda a: (
                        a.name,
                        getattr(a, "alpha_2", getattr(a, "alpha_3", None)),
                    ),
                    list(pycountry.languages),
                )
            ),
        )
    )


def lang_code_to_name(code: str) -> Union[str, None]:
    try:
        return pycountry.languages.lookup(code).name
    except:
        return None


def adjust_contrast_brightness(
    img: np.ndarray, contrast: float = 1.0, brightness: int = 0
):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def has_white(image: np.ndarray):
    # Set RGB values for white
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)

    # Find white pixels within the specified range
    white_pixels = cv2.inRange(image, white_lower, white_upper)

    # Check if any white pixels were found
    return cv2.countNonZero(white_pixels) > 0


display_image_lock = threading.Lock()


def display_image(img: np.ndarray, name: str = "debug"):
    """Show an image in a debug window, or write it to disk when there is no display.

    PySimpleGUI is imported here rather than at module scope on purpose: it pulls in
    tkinter, which is absent from most headless installs. Importing it at the top of
    this module made every entry point unimportable on a server.
    """
    global display_image_lock

    with display_image_lock:
        try:
            import PySimpleGUI as sg

            # Convert the CV2 image array to a format compatible with PySimpleGUI
            image_bytes = cv2.imencode(".png", img)[1].tobytes()

            # Create the GUI layout
            layout = [
                [sg.Text(text=name)],
                [sg.Image(data=image_bytes)],
                [sg.Button("Save"), sg.Button("Close")],
            ]

            # Create the window
            window = sg.Window(name, layout)
        except Exception as e:
            # No tkinter, no PySimpleGUI, or no display attached.
            out_path = os.path.abspath(name + ".png")
            cv2.imwrite(out_path, img)
            print(f"Could not open a debug window ({e}), wrote {name} to {out_path}")
            return

        # Event loop to handle events
        while True:
            event, values = window.read()
            if event == sg.WINDOW_CLOSED or event == "Close":
                break

            if event == "Save":
                cv2.imwrite(name + ".png", img)

        # Close the window
        window.close()


def ensure_gray(img: np.ndarray):
    if len(img.shape) > 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    return img.copy()


def apply_mask(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray, inv=False):
    mask = ensure_gray(mask)
    a_loc, b_loc = foreground.copy(), background.copy()
    mask_inv = cv2.bitwise_not(mask)

    if inv:
        temp = mask
        mask = mask_inv
        mask_inv = temp

    a_loc = cv2.bitwise_and(a_loc, a_loc, mask=mask)
    b_loc = cv2.bitwise_and(b_loc, b_loc, mask=mask_inv)
    return cv2.add(a_loc, b_loc)


def make_bubble_mask(frame: np.ndarray):
    image = frame.copy()
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use the Canny edge detection algorithm
    edges = cv2.Canny(blurred, 50, 150)

    # # Apply morphological closing to close gaps in the circle
    # kernel = np.ones((9, 9), np.uint8)
    # closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # _, binary_image = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Create a black image with the same size as the original

    stage_1 = cv2.drawContours(
        np.zeros_like(image), contours, -1, (255, 255, 255), thickness=2
    )

    stage_1 = cv2.bitwise_not(stage_1)

    stage_1 = cv2.cvtColor(stage_1, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(stage_1, 200, 255, cv2.THRESH_BINARY)

    # Find connected components in the binary image
    num_labels, labels = cv2.connectedComponents(binary_image)

    largest_island_label = np.argmax(np.bincount(labels.flat)[1:]) + 1

    mask = np.zeros_like(image)

    mask[labels == largest_island_label] = 255

    # mask = cv2.bitwise_not(mask)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove black spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return adjust_contrast_brightness(mask, 100)


def get_masked_bounds(mask: np.ndarray):
    gray = ensure_gray(mask)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 200, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    all_contours = []
    for c in contours:
        all_contours.extend(c)
    # display_image(mask,"Test")
    x, y, w, h = cv2.boundingRect(np.array(all_contours))

    return x, y, x + w, y + h


def get_dominant_color(frame: np.ndarray, region_mask=None):
    """The most common BGR colour in a frame, as a (b, g, r) tuple.

    This used to go through cv2.calcHist with 256 bins per channel, which allocates
    67 MB of float32 for every region of every page just to read back one argmax.
    """
    pixels = frame.reshape(-1, frame.shape[-1])

    if region_mask is not None:
        pixels = pixels[ensure_gray(region_mask).reshape(-1) > 0]

    if len(pixels) == 0:
        return 0, 0, 0

    pixels = pixels.astype(np.uint32)
    packed = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    dominant = int(values[counts.argmax()])

    return (dominant >> 16) & 255, (dominant >> 8) & 255, dominant & 255


def mask_text_and_make_bubble_mask(
    frame: np.ndarray, frame_text_mask: np.ndarray, frame_cleaned: np.ndarray
):
    # debug_image(frame_cleaned)
    x1, y1, x2, y2 = get_masked_bounds(frame_text_mask)

    frame_section = frame.copy()[y1:y2, x1:x2]

    mask_section = frame_text_mask.copy()[y1:y2, x1:x2]

    text = apply_mask(
        frame_section,
        np.full(frame_section.shape, 255, dtype=frame_section.dtype),
        mask_section,
    )

    return text, make_bubble_mask(frame_cleaned)


def cv2_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image) -> np.ndarray:
    arr = np.array(img)

    if len(arr.shape) > 2 and arr.shape[2] == 4:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_bounds_for_text(frame_mask: np.ndarray):
    gray = ensure_gray(frame_mask)
    # Threshold the image
    ret, thresh = cv2.threshold(gray, 200, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    largest_contour = max(contours, key=cv2.contourArea)
    polygon = np.array([largest_contour[:, 0, :]])

    rect = lir.lir(polygon)

    return lir.pt1(rect), lir.pt2(rect)


def mask_text_for_in_painting(frame: np.ndarray, mask: np.ndarray):
    image = frame.copy()

    dominant = get_dominant_color(frame)

    # checks if the dominant color is bright or dark with a 0.5 threshold
    is_white = ((sum(dominant) / 3) / 255) > 0.5

    if not is_white:
        image = cv2.bitwise_not(image)

    # Convert the image to grayscale
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
    )  # 15, 5)

    # Perform morphological operations to improve the text extraction
    kernel_size = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    opening = cv2.bitwise_and(opening, opening, mask=ensure_gray(mask))

    # Find contours of the characters
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    new_mask = np.zeros_like(image)

    # Draw contours on the mask
    for contour in contours:
        # Filter out small contours and contours with a large aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        ratio = (w * h) / (len(image) * len(image[0]))
        # print(ratio)
        if ratio < 1:
            # print(ratio)

            cv2.drawContours(
                new_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
            )

            # debug_image(mask,"Segments")

    return new_mask

def in_paint_optimized(
    frame: np.ndarray,
    mask: np.ndarray,
    filtered: list[tuple[tuple[int, int, int, int], str, float]] = [],
    max_height: int = 256,
    max_width: int = 256,
    mask_dilation_kernel_size: int = 9,
    inpaint_fun: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a,
) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = frame.shape
    max_height = int(math.floor(max_height / 8) * 8)
    max_width = int(math.floor(max_width / 8) * 8)

    # only inpaint sections with masks and isolate said masks
    final = frame.copy()
    text_mask = np.zeros_like(mask)

    half_height = int(max_height / 2)
    half_width = int(max_width / 2)

    for bbox, cls, conf in filtered:
        try:
            bx1, by1, bx2, by2 = bbox
            bx1, by1, bx2, by2 = round(bx1), round(by1), round(bx2), round(by2)

            half_bx = round((bx2 - bx1) / 2)
            half_by = round((by2 - by1) / 2)
            midpoint_x, midpoint_y = round(bx1 + half_bx), round(by1 + half_by)

            x1, y1 = max(0, midpoint_x - half_width), max(0, midpoint_y - half_height)

            x2, y2 = min(w, midpoint_x + half_width), min(h, midpoint_y + half_height)

            if y2 < by2:
                y2 = by2

            if y1 > by1:
                y1 = by1

            if x2 < bx2:
                x2 = bx2

            if x1 > bx1:
                x1 = bx1

            # Round the window out to a multiple of 8 by growing it, never by
            # shrinking. Trimming x2/y2 pushed up to 7 rows or columns of the
            # detection box back outside the window, so that strip of text was
            # never inpainted. x2 > x1 and y2 > y1 always hold here (the window
            # is clamped to contain the box), so the old else branches were dead.
            overflow_x = (x2 - x1) % 8
            if overflow_x != 0:
                needed = 8 - overflow_x
                grow_left = min(needed, x1)
                x1 -= grow_left
                x2 = min(w, x2 + (needed - grow_left))

            overflow_y = (y2 - y1) % 8
            if overflow_y != 0:
                needed = 8 - overflow_y
                grow_top = min(needed, y1)
                y1 -= grow_top
                y2 = min(h, y2 + (needed - grow_top))

            bx1 = bx1 - x1
            bx2 = bx2 - x1
            by1 = by1 - y1
            by2 = by2 - y1

            region_mask = mask[y1:y2, x1:x2].copy()

            focus_mask = cv2.rectangle(
                np.zeros_like(region_mask),
                (bx1, by1),
                (bx2, by2),
                (255, 255, 255),
                -1,
            )

            region_mask = apply_mask(
                region_mask, np.zeros_like(region_mask), focus_mask
            )

            if has_white(region_mask):
                (
                    target_region_x1,
                    target_region_y1,
                    target_region_x2,
                    target_region_y2,
                ) = get_masked_bounds(region_mask)

                section_to_in_paint = final[y1:y2, x1:x2]

                section_to_refine = section_to_in_paint[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]
                section_to_refine_mask = region_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]

                # Generate a mask of the actual characters/text
                refined_mask = np.zeros_like(region_mask)
                refined_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = mask_text_for_in_painting(section_to_refine, section_to_refine_mask)

                # The text mask is used for other stuff so we set it here before we dilate for inpainting
                text_mask[y1:y2, x1:x2][
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = refined_mask[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ].copy()

                # Dilate the text mask for inpainting
                kernel = np.ones(
                    (mask_dilation_kernel_size, mask_dilation_kernel_size), np.uint8
                )
                refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)

                # Inpaint using the dilated text mask
                final[y1:y2, x1:x2][
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ] = inpaint_fun(final[y1:y2, x1:x2], refined_mask)[
                    target_region_y1:target_region_y2, target_region_x1:target_region_x2
                ]
        except:
            traceback.print_exc()
            continue

    return final, text_mask


def try_merge_hyphenated(text: list[str], max_chars: int):
    final = []
    total = deque(text)
    current = total.popleft().strip()

    while len(total) > 0 or current != "":
        if (
            len(total) > 0
            and current.endswith("-")
            and len(current[:-1] + total[0]) <= max_chars
        ):
            current = current[:-1] + total.popleft().strip()

        else:
            final.append(current)
            current = total.popleft() if len(total) > 0 else ""

    return final


def wrap_text(text: str, max_chars: int, hyphenator: Union[Hyphenator, None]):
    total = deque(list(filter(lambda a: len(a.strip()) > 0, text.split(" "))))

    if len(total) == 0:
        return []

    current_word = total.popleft()
    lines = []
    current_line = ""
    while len(total) > 0 or len(current_word) > 0:
        sep = " " if len(current_line) > 0 else ""
        new_current = current_line + sep + current_word
        if len(new_current) > max_chars:
            space_left = max_chars - len(current_line + sep)

            try:
                if "-" in current_word:
                    idx = current_word.index("-")
                    total.appendleft(current_word[idx + 1 :])
                    current_word = current_word[:idx]
                    continue
                elif hyphenator is not None:
                    pairs = hyphenator.pairs(current_word)
                else:
                    pairs = []
            except:
                print("EXCEPTION WHEN HYPHENATING:", current_word)
                pairs = []
            if len(pairs) == 0:
                if current_line == "" and len(current_word) > max_chars:
                    return None
                lines.append(current_line)
                current_line = ""
                continue

            pair = min(pairs, key=lambda a: len(current_line + sep + a[0] + "-"))
            if len(current_line + sep + pair[0] + "-") > space_left:
                lines.append(current_line)
                if len(pair[0] + "-") <= max_chars:
                    lines.append(pair[0] + "-")
                    current_line = ""
                    current_word = pair[1]
                    continue
                else:
                    return None

            lines.append(current_line + sep + pair[0] + "-")
            current_line = ""
            current_word = pair[1]
        elif len(total) == 0:
            lines.append(current_line + sep + current_word)
            current_word = ""
        else:
            current_line = new_current
            current_word = total.popleft() if len(total) else ""

    return try_merge_hyphenated(lines, max_chars)


def natural_sort_key(name: str):
    """Sort key that orders page2 before page10 rather than after it."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def reading_order_indices(
    boxes: list[tuple[int, int, int, int]],
    right_to_left: bool = True,
    row_overlap: float = 0.5,
) -> list[int]:
    """Indices of `boxes` in manga reading order: rows top to bottom, and within a
    row right to left.

    Boxes are swept top down and grouped into rows. A box opens a new row once it
    starts more than `row_overlap` of its own height below the bottom of the row
    being built, which keeps side by side bubbles together without merging bubbles
    that merely clip each other's corners.
    """
    if len(boxes) < 2:
        return list(range(len(boxes)))

    def across(index):
        x1, _, x2, _ = boxes[index]
        return -x2 if right_to_left else x1

    order = []
    row = []
    row_bottom = None

    for index in sorted(range(len(boxes)), key=lambda i: (boxes[i][1], across(i))):
        _, y1, _, y2 = boxes[index]

        if row and y1 > row_bottom - (row_overlap * max(1, y2 - y1)):
            order.extend(sorted(row, key=across))
            row = []
            row_bottom = None

        row.append(index)
        row_bottom = y2 if row_bottom is None else max(row_bottom, y2)

    order.extend(sorted(row, key=across))

    return order


def get_fonts() -> list[tuple[str, str]]:
    fonts = []
    for file in filter(lambda a: a.endswith(".ttf"), os.listdir("./fonts")):
        fonts.append((file[0:-4], os.path.abspath(os.path.join("./fonts", file))))

    return fonts


def get_model_path(model=""):
    return os.path.join(os.path.abspath("./models"), model)


def require_model_file(path: str) -> str:
    """Check that a model file exists, failing with download instructions if not."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model weights: {os.path.abspath(path)}\n"
            "Model files are not checked into the repository. Download them with:\n"
            "    ./scripts/fetch_models.sh"
        )

    return path


def get_average_font_size(font: ImageFont, text="some text here"):
    x, y, w, h = font.getbbox(text)
    widths = list(map(lambda a: font.getbbox(a)[2], list(text)))
    widths.sort(reverse=True)
    return widths[1] if len(widths) > 1 else widths[0], h


def get_best_font_size(
    text: str,
    wh: tuple[int, int],
    font_file: str,
    space_between_lines: int = 1,
    start_size: int = 18,
    step: int = 1,
    min_chars_per_line: int = 6,
    initial_iterations: int = 0,
    hyphenator: Union[Hyphenator, None] = None,
) -> Union[tuple[None, None, None, int], tuple[int, int, int, int]]:
    current_font_size = start_size
    current_font = None
    max_width, max_height = wh

    iterations = initial_iterations
    while True:
        iterations += 1

        if current_font_size < 0:
            return None, None, None, iterations

        current_font = ImageFont.truetype(font_file, current_font_size)

        cur_f_width, cur_f_height = get_average_font_size(current_font, text)

        chars_per_line = math.floor(max_width / cur_f_width)

        if chars_per_line < min_chars_per_line:
            current_font_size -= step
            continue

        # print(chars_per_line)
        lines = wrap_text(text, chars_per_line, hyphenator=hyphenator)
        if lines is None:
            current_font_size -= step
            continue

        height_needed = (len(lines) * cur_f_height) + (
            (len(lines) - 1) * space_between_lines
        )
        if height_needed <= max_height:
            return current_font_size, chars_per_line, cur_f_height, iterations
        current_font_size -= step




class COCO_TO_YOLO_TASK:
    SEGMENTATION = "seg"
    DETECTION = "detect"

def resize_and_pad(cv2_image: np.ndarray,target_size: tuple[int,int],extra_padding: int = 0,pad_color: tuple[int,int,int] = (255, 255, 255),interpolation: int = cv2.INTER_CUBIC):
    image = cv2_image.copy()
    height, width = image.shape[:2]
    max_dim = max(height, width)
    image_ratio = width / height
    target_width,target_height = target_size
    target_ratio = target_width / target_height
    should_match_height = target_ratio > image_ratio

    if should_match_height:
        
        height_factor = target_height / height
        new_width = int(height_factor * width)
        image = cv2.resize(image,(new_width,target_height),interpolation=interpolation)
    else:
        width_factor = target_width / width
        new_height = int(width_factor * height)
        image = cv2.resize(image,(target_width,new_height),interpolation=interpolation)

    height, width = image.shape[:2]

    # Calculate the amount of padding needed for each dimension
    pad_height = (target_height - height) + extra_padding
    pad_width = (target_width - width) + extra_padding

    # Determine the amount of padding on each side of the image
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    return cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )
