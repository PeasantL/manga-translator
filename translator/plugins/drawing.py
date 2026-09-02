import cv2
import textwrap
import numpy as np
from PIL import ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
import asyncio
from translator.plugins.base import Drawable, Drawer
from translator.utils import (
    get_best_font_size,
    font_line_height,
    cv2_to_pil,
    pil_to_cv2,
    wrap_to_width,
    drawable_text,
    font_runs,
    fit_box,
    load_font,
)


class HorizontalDrawer(Drawer):
    """Draws text horizontally"""

    # The sizes below are quoted for a page this tall, and scaled to whatever
    # page they are actually drawn on. A bubble covers the same fraction of the
    # page however finely it was scanned, so lettering has to as well: taken as
    # literal pixel counts, one setting is fine print on a 2000 pixel scan and a
    # headline on an 800 pixel one.
    REFERENCE_PAGE_HEIGHT = 1200

    # How far that scaling is allowed to go, so that a thumbnail or a double
    # page spread cannot produce a size nobody can read either way.
    SCALE_RANGE = (0.6, 2.0)

    def __init__(
        self,
        font_file="fonts/animeace2_reg.ttf",
        max_font_size="30",
        line_spacing="2",
        min_font_size="11",
    ) -> None:
        super().__init__()
        self.font_file = font_file
        self.max_font_size = round(float(max_font_size))
        self.line_spacing = round(float(line_spacing))
        self.min_font_size = max(1, round(float(min_font_size)))
        # A maximum under the minimum would leave the search nothing to pick
        # from, and every region would take the overflow path.
        self.max_font_size = max(self.min_font_size, self.max_font_size)

    def sizes_for(self, page_shape) -> tuple[int, int, int]:
        """The smallest size, largest size and line spacing for a page this tall."""
        if page_shape is None:
            return self.min_font_size, self.max_font_size, self.line_spacing

        low, high = self.SCALE_RANGE
        scale = min(high, max(low, page_shape[0] / self.REFERENCE_PAGE_HEIGHT))

        return (
            max(1, round(self.min_font_size * scale)),
            max(1, round(self.max_font_size * scale)),
            max(1, round(self.line_spacing * scale)),
        )

    def box_for(
        self,
        text: str,
        box: tuple[int, int, int, int],
        page_shape: tuple,
        avoid=(),
    ) -> tuple[tuple[int, int, int, int], bool]:
        """Grow the box until the text fits at the minimum readable size.

        `avoid` is what else is being lettered on this page, so that a box which
        has to grow grows over the artwork rather than over another bubble.
        """
        min_size, _, spacing = self.sizes_for(page_shape)

        return fit_box(
            drawable_text(text.strip(), self.font_file),
            box,
            page_shape,
            font_file=self.font_file,
            min_font_size=min_size,
            space_between_lines=spacing,
            hyphenator=Hyphenator("en_US"),
            avoid=avoid,
        )

    # How thick a glyph's border is, as a fraction of the size it is set at, in
    # the two cases that get one. Text over bare artwork carries the heavier of
    # them because there is no telling what is behind it; free text sits on a
    # patch that was cleaned for it and only needs enough to hold its shape
    # against what the cleaner painted back in.
    OUTLINE_RATIO = 6
    BORDER_RATIO = 14

    @classmethod
    def stroke_for(
        cls, font_size: int, should_do_bg: bool, outline: bool, border: bool = False
    ) -> int:
        """How thick an outline this lettering needs.

        Text inside a cleaned bubble takes a hairline in the bubble's own colour
        so that it never quite touches the line art. Text that outgrew its
        bubble is over the drawing itself, and needs a real outline to be read
        against it. Text that never had a bubble is somewhere between the two:
        on artwork, but on artwork chosen to be behind it, so it takes a border
        rather than an outline.
        """
        if outline:
            return max(2, round(font_size / cls.OUTLINE_RATIO))

        if border:
            return max(1, round(font_size / cls.BORDER_RATIO))

        return 2 if should_do_bg else 0

    def layout(
        self,
        text: str,
        frame_w: int,
        frame_h: int,
        hyphenator: Hyphenator,
        min_size: int,
        max_size: int,
        spacing: int,
    ) -> tuple[int, list[str], int]:
        """Font size, wrapped lines and line height for one region.

        Falls back to the minimum size when nothing fits: by this point the box
        has already been grown as far as it is allowed to go, so drawing small
        and overflowing beats drawing nothing.
        """
        font_size, lines, line_height = get_best_font_size(
            text,
            (frame_w, frame_h),
            font_file=self.font_file,
            space_between_lines=spacing,
            start_size=max_size,
            hyphenator=hyphenator,
            min_size=min_size,
        )

        if font_size:
            return font_size, lines, line_height

        font = load_font(self.font_file, min_size)
        lines = wrap_to_width(font, text, frame_w, hyphenator=hyphenator)

        # No wrap fits the width, which means a single word wider than the box.
        # textwrap will break it mid-word, the only way to place it at all.
        if lines is None:
            advance = font.getlength(text) / len(text)
            lines = textwrap.wrap(text, max(1, int(frame_w // max(1.0, advance))))

        return min_size, lines or [text], font_line_height(font)

    async def draw(
        self,batch: list[Drawable]
    ) -> list[tuple[ndarray,ndarray]]:
        return await asyncio.gather(*[self.draw_one(x) for x in batch])
                
    
    async def draw_one(
        self, item: Drawable
    ) -> tuple[ndarray,ndarray]:
        item_mask = np.zeros_like(item.frame)

        # Characters no font on disk can draw are dropped here, before sizing, so
        # that the layout is measured on what actually gets drawn.
        text = drawable_text(item.translation.text.strip(), self.font_file)

        if len(text) <= 0:
            return (item.frame,item_mask)

        frame_h, frame_w, _ = item.frame.shape

        hyphenator = Hyphenator("en_US")
        min_size, max_size, spacing = self.sizes_for(item.page_shape)

        color_fg, color_bg, should_do_bg = item.color

        # Text that outgrew its bubble is over whatever the bubble was sitting
        # on. Over line art an outline in the bubble's own colour is enough to
        # read it against, and it leaves the drawing intact; over something
        # solid - a black panel, a screentone - nothing shows through it, so it
        # gets a panel of its own.
        busy = False

        if item.backdrop:
            darkness = float((item.frame.mean(axis=2) < 128).mean())
            busy = darkness >= 0.25

        outline = item.backdrop and not busy
        # Free text is bordered at whatever size it came out at, fitting or
        # not. Not folded into `outline`: that one is heavy enough to be read
        # over an untouched drawing, and this text is sitting on a patch the
        # cleaner painted for it.
        border = item.outline and not outline

        font_size, wrapped, line_height = self.layout(
            text, frame_w, frame_h, hyphenator, min_size, max_size, spacing
        )

        if len(wrapped) == 0:
            return (item.frame,item_mask)

        # A stroke is drawn around every glyph, so it costs the block its own
        # width on all four sides. Laid out once to find out how thick it will
        # be, then again inside what that leaves - a layout measured without it
        # is exactly a stroke too big for the room it has.
        stroke_width = self.stroke_for(font_size, should_do_bg, outline, border)

        if stroke_width > 0:
            font_size, wrapped, line_height = self.layout(
                text,
                max(1, frame_w - (2 * stroke_width)),
                max(1, frame_h - (2 * stroke_width)),
                hyphenator,
                min_size,
                max_size,
                spacing,
            )

        stroke_color = color_bg

        frame_as_pil = cv2_to_pil(item.frame)
        
        mask_as_pil = cv2_to_pil(item_mask)

        image_draw = ImageDraw.Draw(frame_as_pil)

        mask_draw = ImageDraw.Draw(mask_as_pil)

        # A line is usually one run in the chosen font. It is split only where
        # that font has no glyph, so a symbol it lacks is drawn from a font that
        # has it instead of coming out as an empty box.
        lines = [font_runs(line, self.font_file, font_size) for line in wrapped]
        line_widths = [
            sum(font.getlength(part) for part, font in runs) for runs in lines
        ]

        block_height = (len(wrapped) * line_height) + (
            (len(wrapped) - 1) * spacing
        )
        block_top = (frame_h - block_height) / 2

        if busy:
            padding = max(4, round(font_size / 3))
            block_width = max(line_widths) if len(line_widths) > 0 else 0
            panel = (
                max(0, ((frame_w - block_width) / 2) - padding),
                max(0, block_top - padding),
                min(frame_w, ((frame_w + block_width) / 2) + padding),
                min(frame_h, block_top + block_height + padding),
            )

            # The panel is the bubble's own background, so white on black
            # lettering stays white on black once it leaves the bubble.
            image_draw.rounded_rectangle(
                panel, radius=padding, fill=(*color_bg, 255)
            )
            mask_draw.rounded_rectangle(
                panel, radius=padding, fill=(255, 255, 255, 255)
            )

        for line_no, runs in enumerate(lines):
            line_width = line_widths[line_no]

            draw_y = block_top + (line_no * (line_height + spacing))

            # Not abs(): a line wider than its box is centred on the box and
            # hangs off both ends, which is a line that reads. Taken as a
            # positive number it was pushed right instead, and the end of it
            # went off the edge of the frame and was never drawn.
            draw_x = (frame_w - line_width) / 2

            for part, font in runs:
                image_draw.text(
                    (draw_x, draw_y),
                    part,
                    fill=(*color_fg,255),
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=(*stroke_color,255) if stroke_width > 0 else None
                )

                mask_draw.text(
                    (draw_x, draw_y),
                    part,
                    fill=(255, 255, 255, 255),
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=(255, 255, 255) if stroke_width > 0 else None
                )

                draw_x += font.getlength(part)

        mask_cv2 = cv2.cvtColor(pil_to_cv2(mask_as_pil),cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(mask_cv2, 1, 255, cv2.THRESH_BINARY)

        return (pil_to_cv2(frame_as_pil),binary_mask)


    @staticmethod
    def get_name() -> str:
        return "Horizontal Drawer"
