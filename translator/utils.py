import cv2
import os
import re
import math
import torch
import threading
import numpy as np
import largestinteriorrectangle as lir
from typing import Union, Callable
from PIL import Image, ImageFont
from hyphen import Hyphenator
from collections import deque
from functools import lru_cache
import traceback


class TranslatorGlobals:
    COLOR_BLACK = np.array((0, 0, 0))
    COLOR_WHITE = np.array((255, 255, 255))

def get_torch_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

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


def luma(color) -> float:
    """Perceived brightness of a BGR colour, 0 to 255."""
    blue, green, red = (float(v) for v in color[:3])

    return (0.114 * blue) + (0.587 * green) + (0.299 * red)


def measure_region_colors(
    frame: np.ndarray, frame_clean: np.ndarray, text_mask: np.ndarray, spread: int = 7
):
    """The colour of a bubble's text, and of what the text was drawn on.

    Both are measured rather than predicted, off the refined per glyph mask the
    cleaner built: the background from the cleaned page, where the text has
    already been erased, and the text from the same pixels in the original.

    Returns (None, None) when there are no glyph pixels to measure.
    """
    mask = ensure_gray(text_mask)
    glyphs = mask > 0

    if not glyphs.any():
        return None, None

    # Just the area under and immediately around the glyphs, so a box that
    # catches some of the panel outside the bubble is not measured as if the
    # text sat on it.
    kernel = np.ones((spread, spread), np.uint8)
    around = cv2.dilate(mask, kernel, iterations=1) > 0
    background = frame_clean[around] if around.any() else frame_clean.reshape(-1, 3)

    background_color = tuple(int(v) for v in np.median(background, axis=0))

    pixels = frame[glyphs]
    brightness = pixels @ np.array((0.114, 0.587, 0.299))

    # The mask hugs the glyph outline, so a good half of what it covers is the
    # antialiased edge, and the median of that is a grey which is neither the
    # lettering nor the paper. Measure the end of the range the lettering is on
    # instead - the darkest tenth for dark text, the lightest tenth for light.
    if np.median(brightness) < luma(background_color):
        core = pixels[brightness <= np.percentile(brightness, 10)]
    else:
        core = pixels[brightness >= np.percentile(brightness, 90)]

    text_color = tuple(int(v) for v in np.median(core, axis=0))

    return text_color, background_color


def drawing_colors(text_color, background_color):
    """What to draw a region with: foreground, outline colour, and whether to outline.

    Manga lettering is black or white far more often than it is anything else,
    and the median of an antialiased glyph lands short of both, so a measurement
    near either end is snapped to it. A measurement that would leave the text
    barely distinguishable from what it sits on is discarded in favour of
    whichever of black or white the background is not - being wrong about the
    shade is recoverable, drawing black on black is not.
    """
    background = np.array(
        background_color if background_color is not None else (255, 255, 255)
    )
    background_luma = luma(background)

    if text_color is None:
        foreground = (
            TranslatorGlobals.COLOR_WHITE
            if background_luma < 128
            else TranslatorGlobals.COLOR_BLACK
        )
    else:
        foreground = np.array(text_color)

        if luma(foreground) < 90:
            foreground = TranslatorGlobals.COLOR_BLACK
        elif luma(foreground) > 165:
            foreground = TranslatorGlobals.COLOR_WHITE

        if abs(luma(foreground) - background_luma) < 60:
            foreground = (
                TranslatorGlobals.COLOR_WHITE
                if background_luma < 128
                else TranslatorGlobals.COLOR_BLACK
            )

    # Light text on a dark bubble is outlined in the bubble's own colour, so it
    # stays readable where the box overhangs the bubble by a pixel or two.
    return foreground, background, luma(foreground) > background_luma


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


FALLBACK_FONTS = (
    "NotoSansJP-Regular.ttf",
    "msmincho.ttf",
    "reiko.ttf",
)


@lru_cache(maxsize=32)
def font_charset(font_file: str) -> frozenset:
    """The code points a font can actually draw.

    Pillow has no fallback: a character the font has no glyph for is drawn as
    .notdef, the empty box readers see as []. Comic fonts carry no symbols, so a
    heart in a translation comes out as a box. Reading the cmap is the only way
    to know before drawing.
    """
    try:
        from fontTools.ttLib import TTFont

        with TTFont(font_file, fontNumber=0, lazy=True) as font:
            return frozenset(font.getBestCmap().keys())
    except Exception:
        # Better to assume the font covers everything than to drop text because
        # its cmap could not be parsed.
        return frozenset()


def font_for_char(char: str, font_file: str) -> Union[str, None]:
    """Which font file to draw one character with, or None if nothing can.

    The chosen font is preferred whenever it has the glyph, so ordinary text is
    never split up. Only what it lacks goes looking through the fallbacks.
    """
    code = ord(char)
    charset = font_charset(font_file)

    if len(charset) == 0 or code in charset:
        return font_file

    for name in FALLBACK_FONTS:
        candidate = os.path.abspath(os.path.join("./fonts", name))

        if os.path.isfile(candidate) and code in font_charset(candidate):
            return candidate

    return None


def drawable_text(text: str, font_file: str) -> str:
    """Drop the characters no available font can draw.

    An emoji no font on disk has is better left out than drawn as a box. Exotic
    whitespace - an ideographic space carried over from the source - becomes a
    plain space, which every font has and which wrap_text can break on.
    """
    return "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or font_for_char(char, font_file) is not None
    )


def font_runs(text: str, font_file: str, size: int) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    """Split a line into runs, each with the font that can draw it.

    Text the chosen font covers comes back as a single run, which is the usual
    case; a heart in the middle of a sentence comes back as three.
    """
    runs: list[tuple[str, ImageFont.FreeTypeFont]] = []
    current = ""
    current_file = None

    for char in text:
        chosen = font_for_char(char, font_file)

        if chosen is None:
            continue

        if chosen != current_file:
            if len(current) > 0:
                runs.append((current, load_font(current_file, size)))

            current, current_file = "", chosen

        current += char

    if len(current) > 0:
        runs.append((current, load_font(current_file, size)))

    return runs


@lru_cache(maxsize=64)
def load_font(font_file: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_file, size)


def get_model_path(model=""):
    return os.path.join(os.path.abspath("./models"), model)


def require_model_file(path: str) -> str:
    """Check that a model file exists, failing with download instructions if not."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model weights: {os.path.abspath(path)}\n"
            "Model files are not checked into the repository. Download them with:\n"
            "    ./fetch_models.sh"
        )

    return path


def font_line_height(font: ImageFont.FreeTypeFont) -> int:
    """How much vertical room one line of this font needs.

    Taken from the font's own metrics rather than the bounding box of the text
    being drawn, so that a line of "no" and a line of "Why?!" are given the same
    height and a block of them comes out evenly spaced.
    """
    ascent, descent = font.getmetrics()

    return ascent + descent


def wrap_to_width(
    font: ImageFont.FreeTypeFont,
    text: str,
    max_width: int,
    hyphenator: Union[Hyphenator, None] = None,
    min_chars_per_line: int = 6,
) -> Union[list[str], None]:
    """Wrap `text` into lines that really are no wider than `max_width`.

    wrap_text counts characters, so it has to be given a character budget rather
    than a width. The first guess is what a character of this text costs on
    average in this font; the wrap it produces is then measured and the budget
    tightened until every line genuinely fits.

    The guess used to be the second widest character in the string, applied to
    every character in it. That is where most of "text comes out too small" came
    from: a proportional font's widest character runs to twice its average, so
    every line was budgeted about half the text it could hold, and the search
    below kept shrinking the font to make room that was already there.
    """
    if len(text) == 0:
        return []

    advance = font.getlength(text) / len(text)
    chars = int(max_width // advance) if advance > 0 else 0

    while chars >= min_chars_per_line:
        lines = wrap_text(text, chars, hyphenator=hyphenator)

        # A word too long for the line even on its own. A tighter budget only
        # makes that worse, so there is nothing to retry.
        if lines is None:
            return None

        if all(font.getlength(line) <= max_width for line in lines):
            return lines

        chars -= 1

    return None


def get_best_font_size(
    text: str,
    wh: tuple[int, int],
    font_file: str,
    space_between_lines: int = 1,
    start_size: int = 18,
    min_chars_per_line: int = 6,
    hyphenator: Union[Hyphenator, None] = None,
    min_size: int = 1,
) -> Union[tuple[None, None, None], tuple[int, list[str], int]]:
    """The largest size from min_size to start_size whose text fits `wh`.

    Returns the size, the lines it wraps into at that size, and the height of
    one line - the lines included because the caller would otherwise have to
    reproduce the wrap, and any difference in how it did so would be drawn.

    (None, None, None) means the text does not fit even at the minimum, which is
    the caller's cue to grow the box rather than shrink the text further.

    Binary searched rather than stepped down a pixel at a time: text that fits
    at one size fits at every smaller one, and a chapter is hundreds of these.
    """
    max_width, max_height = wh

    def attempt(size: int) -> Union[tuple[list[str], int], None]:
        font = load_font(font_file, size)
        lines = wrap_to_width(
            font, text, max_width, hyphenator=hyphenator,
            min_chars_per_line=min_chars_per_line,
        )

        if lines is None or len(lines) == 0:
            return None

        line_height = font_line_height(font)
        needed = (len(lines) * line_height) + (
            (len(lines) - 1) * space_between_lines
        )

        return (lines, line_height) if needed <= max_height else None

    low, high = min_size, start_size
    best = None

    while low <= high:
        size = (low + high) // 2
        fitted = attempt(size)

        if fitted is None:
            high = size - 1
        else:
            best = (size, fitted[0], fitted[1])
            low = size + 1

    return best if best is not None else (None, None, None)


def fit_box(
    text: str,
    box: tuple[int, int, int, int],
    page_shape: tuple,
    font_file: str,
    min_font_size: int,
    space_between_lines: int = 2,
    hyphenator: Union[Hyphenator, None] = None,
    max_scale: float = 2.5,
) -> tuple[tuple[int, int, int, int], bool]:
    """How much room this text needs, and whether that is more than it was given.

    A long translation in a small bubble used to be shrunk until it fit, which
    at some point means unreadable, or dropped entirely when even that failed.
    Instead the box is grown around its own centre until the text fits at the
    minimum readable size. The caller is told when that happened, so the text
    can be drawn on a backdrop - it is now over artwork, not over a cleaned
    bubble.
    """
    page_height, page_width = page_shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = x2 - x1, y2 - y1

    if width <= 0 or height <= 0 or len(text.strip()) == 0:
        return (x1, y1, x2, y2), False

    def fits(w: int, h: int) -> bool:
        size, _, _ = get_best_font_size(
            text,
            (w, h),
            font_file=font_file,
            space_between_lines=space_between_lines,
            start_size=min_font_size,
            hyphenator=hyphenator,
            min_size=min_font_size,
        )

        return size is not None

    if fits(width, height):
        return (x1, y1, x2, y2), False

    centre_x, centre_y = x1 + (width / 2), y1 + (height / 2)
    grown = (x1, y1, x2, y2)
    scale = 1.0

    while scale < max_scale:
        scale += 0.1

        new_width = min(page_width, round(width * scale))
        new_height = min(page_height, round(height * scale))

        # Keep the text over the bubble it came from, but never off the page.
        new_x1 = int(max(0, min(page_width - new_width, round(centre_x - (new_width / 2)))))
        new_y1 = int(max(0, min(page_height - new_height, round(centre_y - (new_height / 2)))))

        grown = (new_x1, new_y1, new_x1 + new_width, new_y1 + new_height)

        if fits(new_width, new_height):
            return grown, True

        if new_width == page_width and new_height == page_height:
            break

    # Nothing fit even at the largest allowed size. Hand back the biggest box
    # tried anyway; the drawer draws at the minimum size and overflows it, which
    # is still more use than an empty bubble.
    return grown, True


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


def read_image(path: str) -> Union[np.ndarray, None]:
    """cv2.imread that works with non ASCII paths.

    On Windows cv2.imread hands the path to the ANSI file API, so a folder named
    with anything outside the local code page - a chapter title with a star or a
    Japanese character in it - simply fails to open. Reading the bytes ourselves
    and decoding them in memory sidesteps the path entirely.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# The encoder setting each format calls "quality". Only the lossy ones have
# one; PNG's compression level is a speed/size trade rather than a fidelity
# one, so it is deliberately not reachable through the same argument.
_QUALITY_FLAG = {
    ".webp": cv2.IMWRITE_WEBP_QUALITY,
    ".jpg": cv2.IMWRITE_JPEG_QUALITY,
    ".jpeg": cv2.IMWRITE_JPEG_QUALITY,
}


def write_image(path: str, frame: np.ndarray, quality: Union[int, None] = None) -> bool:
    """cv2.imwrite that works with non ASCII paths, and that says when it failed.

    cv2.imwrite returns False rather than raising, so a whole chapter could be
    reported as written while nothing reached the disk.

    `quality` is the encoder's own scale for whatever format the extension asks
    for -- WebP and JPEG both take 0 to 100, and WebP reads anything above 100
    as a request for lossless. It is ignored for a format that has no such
    setting, so a caller can pass one without first knowing what it is writing.
    """
    extension = os.path.splitext(path)[1] or ".png"
    params = []

    if quality is not None and (flag := _QUALITY_FLAG.get(extension.lower())):
        params = [flag, int(quality)]

    success, encoded = cv2.imencode(extension, frame, params)

    if not success:
        return False

    try:
        encoded.tofile(path)
    except OSError:
        return False

    return True
