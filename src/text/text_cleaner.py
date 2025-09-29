from . import cantonese
from . import chinese2
from . import english
from . import japanese
from . import korean
from ..symbols import symbols
from typing import List, Tuple, Optional

language_module_map = {
    "zh": chinese2,
    "ja": japanese,
    "en": english,
    "ko": korean,
    "yue": cantonese,
}


def load_language_module(language):
    mod = language_module_map[language]
    text_normalize = getattr(mod, "text_normalize", None)
    g2p = getattr(mod, "g2p", None)
    if g2p is None:
        raise NotImplementedError(f"{language} g2p not found!")
    return text_normalize, g2p


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
    Returns:
        List of integers corresponding to the symbols in the text
    """
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text: str, language: str) -> Tuple[List[str], Optional[List[int]], str]:
    if language not in language_module_map:
        language = "en"
        text = " "
    text_normalize, g2p = load_language_module(language)
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, special_s, target_symbol, text_normalize, g2p)
    if text_normalize:
        norm_text = text_normalize(text)
    else:
        norm_text = text
    if language == "zh" or language == "yue":  ##########
        phones, word2ph = g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones = g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, special_s, target_symbol, text_normalize, g2p):
    text = text.replace(special_s, ",")
    if text_normalize is None:
        raise ValueError("text_normalize is None!")
    norm_text = text_normalize(text)
    phones = g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def clean_text_inf(text: str, language: str):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text
