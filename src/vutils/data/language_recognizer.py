import re
from enum import Enum
from typing import List, Optional


class Language(Enum):
    ENGLISH = r"[A-Za-z\s]"
    CHINESE = r"[\u4e00-\u9fff]"
    # JAPANESE = 'ja'
    # RUSSIAN = 'ru'
    # FRENCH = 'fr'
    # GERMAN = 'de'
    # SPANISH = 'es'
    # ITALIAN = 'it'
    # KOREAN = 'ko'


class Mode(Enum):
    EXISTS = 'exists'
    ALL = 'all'


class Recognizer:
    def __init__(self, options: List[Language] = [Language.ENGLISH, Language.CHINESE]):
        self.options = options

    def recognize(self, text: str, mode: Mode = Mode.EXISTS) -> Optional[Language]:
        if mode == Mode.ALL:
            recognize_fct = self.__recognize_all
        elif mode == Mode.EXISTS:
            recognize_fct = self.__recognize_exists
        else:
            raise ValueError(f'Unrecognized mode: {mode}')
        for option in self.options:
            if recognize_fct(option.value, text):
                return option
        return None

    def __recognize_exists(self, pattern: str, text: str) -> bool:
        if re.search(pattern, text):
            return True
        else:
            return False

    def __recognize_all(self, pattern: str, text: str) -> bool:
        if re.sub(pattern, '', text).strip() == '':
            return True
        else:
            return False
