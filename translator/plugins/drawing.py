import cv2
import math
import textwrap
import numpy as np
from PIL import ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
import asyncio
from translator.plugins.base import (
    Drawable,
    Drawer,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)
from translator.utils import (
    get_best_font_size,
    get_average_font_size,
    cv2_to_pil,
    pil_to_cv2,
    wrap_text,
    get_fonts,
    drawable_text,
    font_runs,
    fit_box,
    load_font,
)


class HorizontalDrawer(Drawer):
    """Draws text horizontally"""

    HYPHENATE_LAST = "last"
    HYPHENATE_ALWAYS = "always"

    # How far a box may grow to take the text. Growing further is allowed while
    # trying to keep words whole, because that is the whole point of asking for
    # whole words - a wider panel over the art beats a word in three pieces.
    MAX_GROWTH = 2.5
    MAX_GROWTH_WHOLE_WORDS = 3.5

    def __init__(
        self,
        font_file="fonts/animeace2_reg.ttf",
        max_font_size="13",
        line_spacing="2",
        min_font_size="13",
        hyphenate="last",
    ) -> None:
        super().__init__()
        self.font_file = font_file
        self.max_font_size = round(float(max_font_size))
        self.line_spacing = round(float(line_spacing))
        self.min_font_size = max(1, round(float(min_font_size)))
        self.hyphenate = str(hyphenate).strip().lower()

    def avoids_hyphens(self) -> bool:
        return self.hyphenate != HorizontalDrawer.HYPHENATE_ALWAYS

    def box_for(
        self, text: str, box: tuple[int, int, int, int], page_shape: tuple
    ) -> tuple[tuple[int, int, int, int], bool]:
        """Grow the box until the text fits at the minimum readable size.

        With hyphenation held back, the first thing tried is a box that takes
        the text in whole words. Only if no allowed size of box can do that is
        the word broken - so a translation one letter too wide for its bubble
        spills out of it instead of being split.
        """
        text = drawable_text(text.strip(), self.font_file)
        hyphenator = Hyphenator("en_US")

        common = dict(
            box=box,
            page_shape=page_shape,
            font_file=self.font_file,
            min_font_size=self.min_font_size,
            space_between_lines=self.line_spacing,
        )

        if self.avoids_hyphens():
            whole, expanded, fitted = fit_box(
                text,
                hyphenator=None,
                max_scale=HorizontalDrawer.MAX_GROWTH_WHOLE_WORDS,
                **common,
            )

            if fitted:
                return whole, expanded

        grown, expanded, _ = fit_box(
            text,
            hyphenator=hyphenator,
            max_scale=HorizontalDrawer.MAX_GROWTH,
            **common,
        )

        return grown, expanded

    def layout(
        self, text: str, frame_w: int, frame_h: int, hyphenator: Hyphenator
    ) -> tuple[int, list[str], int]:
        """Font size, wrapped lines and line height for one region.

        Sized without hyphenation first, so the largest size that keeps every
        word whole wins over a larger one that would break them. box_for has
        already grown the box to make that possible where it could, so this
        usually succeeds on the first attempt.

        Falls back to the minimum size when nothing fits: by that point the box
        is as large as it is allowed to get, so drawing small and overflowing
        beats drawing nothing.
        """
        attempts = [None, hyphenator] if self.avoids_hyphens() else [hyphenator]

        for attempt in attempts:
            font_size, chars_per_line, line_height, _ = get_best_font_size(
                text,
                (frame_w, frame_h),
                font_file=self.font_file,
                space_between_lines=self.line_spacing,
                start_size=self.max_font_size,
                step=1,
                hyphenator=attempt,
                min_size=self.min_font_size,
            )

            if font_size:
                return (
                    font_size,
                    wrap_text(text, chars_per_line, hyphenator=attempt),
                    line_height,
                )

        font_size = self.min_font_size
        char_width, line_height = get_average_font_size(
            load_font(self.font_file, font_size), text
        )
        chars_per_line = max(1, math.floor(frame_w / max(1, char_width)))

        # wrap_text gives up on a word longer than the line; textwrap will break
        # it, which is the only way to place a very long word in a narrow box.
        wrapped = wrap_text(text, chars_per_line, hyphenator=hyphenator)

        if wrapped is None:
            wrapped = textwrap.wrap(text, chars_per_line) or [text]

        return font_size, wrapped, line_height

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

        font_size, wrapped, line_height = self.layout(
            text, frame_w, frame_h, hyphenator
        )

        if len(wrapped) == 0:
            return (item.frame,item_mask)

        frame_as_pil = cv2_to_pil(item.frame)
        
        mask_as_pil = cv2_to_pil(item_mask)

        image_draw = ImageDraw.Draw(frame_as_pil)

        mask_draw = ImageDraw.Draw(mask_as_pil)
        color_fg, color_bg, should_do_bg = item.color

        stroke_width = 2 if should_do_bg else 0
        stroke_color = color_bg

        # Text that outgrew its bubble is over whatever the bubble was sitting
        # on. Over line art an outline in the bubble's own colour is enough to
        # read it against, and it leaves the drawing intact; over something
        # solid - a black panel, a screentone - nothing shows through it, so it
        # gets a panel of its own.
        busy = False

        if item.backdrop:
            darkness = float((item.frame.mean(axis=2) < 128).mean())
            busy = darkness >= 0.25

            if not busy:
                stroke_width = max(stroke_width, max(2, round(font_size / 6)))

        # A line is usually one run in the chosen font. It is split only where
        # that font has no glyph, so a symbol it lacks is drawn from a font that
        # has it instead of coming out as an empty box.
        lines = [font_runs(line, self.font_file, font_size) for line in wrapped]
        line_widths = [
            sum(font.getlength(part) for part, font in runs) for runs in lines
        ]

        block_top = self.line_spacing + (
            (
                frame_h
                - ((len(wrapped) * line_height) + (len(wrapped) * self.line_spacing))
            )
            / 2
        )

        if busy:
            padding = max(4, round(font_size / 3))
            block_width = max(line_widths) if len(line_widths) > 0 else 0
            block_height = (len(wrapped) * line_height) + (
                (len(wrapped) - 1) * self.line_spacing
            )
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

            draw_y = (
                block_top
                + (line_no * line_height)
                + (self.line_spacing * line_no)
            )

            draw_x = abs((frame_w - line_width) / 2)

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
    def get_arguments() -> list[PluginArgument]:
        fonts_available = get_fonts()
        return [
            PluginSelectArgument(
                id="font_file",
                name="Font",
                description="The font to draw with",
                options=list(
                    map(
                        lambda a: PluginSelectArgumentOption(name=a[0], value=a[1]),
                        fonts_available,
                    )
                ),
                default=fonts_available[0][1],
            ),
            PluginArgument(
                id="max_font_size",
                name="Max Font Size",
                description="The size text starts at, shrinking from there until "
                "it fits. Set equal to the minimum for one size everywhere",
                default="13",
            ),
            PluginArgument(
                id="line_spacing",
                name="Line Spacing",
                description="Space between lines",
                default="2",
            ),
            PluginSelectArgument(
                id="hyphenate",
                name="Break Words",
                description="Whether a word may be split across lines. Held back "
                "to a last resort, the box grows past the bubble to keep words "
                "whole and the text is drawn on white where it overhangs",
                options=[
                    PluginSelectArgumentOption(
                        "Only when nothing else fits",
                        HorizontalDrawer.HYPHENATE_LAST,
                    ),
                    PluginSelectArgumentOption(
                        "Whenever it fills the line",
                        HorizontalDrawer.HYPHENATE_ALWAYS,
                    ),
                ],
                default=HorizontalDrawer.HYPHENATE_LAST,
            ),
            PluginArgument(
                id="min_font_size",
                name="Min Font Size",
                description="The smallest readable font size. Text that will not "
                "fit its bubble at this size spills past it at this size, with "
                "white behind it, instead of being shrunk further",
                default="13",
            ),
        ]

    @staticmethod
    def get_name() -> str:
        return "Horizontal Drawer"
