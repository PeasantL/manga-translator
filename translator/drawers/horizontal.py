import cv2
import numpy as np
from PIL import ImageDraw
from numpy import ndarray
from hyphen import Hyphenator
import asyncio
from translator.core.plugin import (
    Drawable,
    Drawer,
    PluginArgument,
    PluginSelectArgument,
    PluginSelectArgumentOption,
)
from translator.utils import (
    get_best_font_size,
    cv2_to_pil,
    pil_to_cv2,
    wrap_text,
    get_fonts,
    drawable_text,
    font_runs,
)


class HorizontalDrawer(Drawer):
    """Draws text horizontally"""

    def __init__(
        self, font_file="fonts/animeace2_reg.ttf", max_font_size="30", line_spacing="2"
    ) -> None:
        super().__init__()
        self.font_file = font_file
        self.max_font_size = round(float(max_font_size))
        self.line_spacing = round(float(line_spacing))

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

        font_size, chars_per_line, line_height, iters = get_best_font_size(
            text,
            (frame_w, frame_h),
            font_file=self.font_file,
            space_between_lines=self.line_spacing,
            start_size=self.max_font_size,
            step=1,
            hyphenator=hyphenator,
        )

        if not font_size:
            return (item.frame,item_mask)

        wrapped = wrap_text(text, chars_per_line, hyphenator=hyphenator)

        frame_as_pil = cv2_to_pil(item.frame)
        
        mask_as_pil = cv2_to_pil(item_mask)

        image_draw = ImageDraw.Draw(frame_as_pil)

        mask_draw = ImageDraw.Draw(mask_as_pil)
        color_fg, color_bg, should_do_bg = item.color

        stroke_width = 2 if should_do_bg else 0

        for line_no in range(len(wrapped)):
            line = wrapped[line_no]

            # A line is usually one run in the chosen font. It is split only
            # where that font has no glyph, so a symbol it lacks is drawn from a
            # font that has it instead of coming out as an empty box.
            runs = font_runs(line, self.font_file, font_size)
            line_width = sum(font.getlength(part) for part, font in runs)

            draw_y = (
                self.line_spacing
                + (
                    (
                        frame_h
                        - (
                            (len(wrapped) * line_height)
                            + (len(wrapped) * self.line_spacing)
                        )
                    )
                    / 2
                )
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
                    stroke_fill=(*color_bg,255) if stroke_width > 0 else None
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
                description="The max font size for the sizing algorithm",
                default="30",
            ),
            PluginArgument(
                id="line_spacing",
                name="Line Spacing",
                description="Space between lines",
                default="2",
            ),
        ]

    @staticmethod
    def get_name() -> str:
        return "Horizontal Drawer"
