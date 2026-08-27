from translator.core.plugin import Cleaner
from translator.cleaners.lama import LamaCleaner


def get_cleaners() -> list[Cleaner]:
    return [LamaCleaner]
