from typing import List
from spacy.language import Language


def parse_reference(text: str, spacy_model: Language) -> List[str]:
    """
    Extract morphological features of the antecedents

    Parameters
    ----------
    text: str
        anaphor
    spacy_model: spacy.language.Language
        spacy model to be used for morphological analysis

    Returns
    -------
    List[str]
        Number and/or Gender of the anaphor parameter strings for Matcher
    """
    out = {}
    morph = spacy_model(text)[0].morph
    case = morph.get("Case")
    if len(case) > 0:
        case = case[0]
        out["case"] = f"Case={case}"
    gender = morph.get("Gender")
    if len(gender) > 0:
        gender = gender[0]
        out["gender"] = f"Gender={gender}"
    number = morph.get("Number")
    if len(number) > 0:
        number = number[0]
        out["number"] = f"Number={number}"
    return out
