from translator.core.plugin import Drawer
from translator.drawers.horizontal import HorizontalDrawer


def get_drawers() -> list[Drawer]:
    return list(filter(lambda a: a.is_valid(), [HorizontalDrawer]))
