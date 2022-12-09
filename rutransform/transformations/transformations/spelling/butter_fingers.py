import itertools
import random
import spacy
from typing import List, Optional, Union
from spacy.language import Language

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

"""
Adapted from https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation
"""


def butter_finger(
    text: str,
    spacy_model: Language,
    prob: float = 0.1,
    seed: int = 42,
    max_outputs: int = 1,
    stop_words: List[int] = None,
) -> List[str]:
    """
    Adds typos to text the sentence using keyboard distance

    Parameters
    ----------
    text: str
        text to transform
    spacy_model: spacy.language.Language
        spacy model used for lemmatization
    prob: float
        probability of the transformation (default is 0.1)
    seed: int
        seed to freeze everything (default is 42)
    max_outputs: int
        maximum number of the returned sentences (default is 1)
    stop_words: List[int], optional
        stop words to ignore during transformation (default is None)

    Returns
    -------
    List[str]
        list of transformed sentences
    """
    random.seed(seed)
    key_approx = {
        "й": "йцфыувяч",
        "ц": "цйуыфвкасч",
        "у": "уцкыавйфячсмпе",
        "к": "куевпацычсмпе",
        "е": "екнарпувсмитог",
        "н": "негпоркамитош",
        "г": "гншрлоепитьдщ",
        "ш": "шгщодлнртьдз",
        "щ": "щшзлдгоь",
        "з": "здщхэшл",
        "х": "хзъэж\щдю.",
        "ъ": "ъх\зэж.",
        "ф": "фйыяцчцвсу",
        "ы": "ыцчфвкам",
        "в": "вусыафйпим",
        "а": "авпкмцычнрт",
        "п": "пеиарувснот",
        "р": "рнтпоакмлшь",
        "о": "орлтгпеидщь",
        "л": "лодштнрт",
        "д": "дщльзгот",
        "ж": "жз.дэх\ю",
        "э": "эхж\зъ.",
        "я": "яфчымву",
        "ч": "чясывимакуцй",
        "с": "счмваяыцукпи",
        "м": "мсаипчвукент",
        "и": "имтпрсаенгт",
        "т": "тиьромпегшл",
        "ь": "ьтлодщшл",
        "б": "блдьюож",
        "ю": "юджб.ьл",
        " ": " ",
    }
    if stop_words is None:
        stop_words = []

    transformed_texts = []
    split_text = [token.text for token in spacy_model(text)]
    for _ in itertools.repeat(None, max_outputs):
        butter_text = []
        for w, word in enumerate(split_text):
            if w in stop_words:
                butter_text.append(word)
            else:
                new_word = ""
                for letter in word:
                    lcletter = letter.lower()
                    if lcletter not in key_approx.keys():
                        new_letter = lcletter
                    else:
                        if random.uniform(0, 1) <= prob:
                            new_letter = random.choice(key_approx[lcletter])
                        else:
                            new_letter = lcletter
                    # go back to original case
                    if not lcletter == letter:
                        new_letter = new_letter.upper()
                    new_word += new_letter
                butter_text.append(new_word)
        transformed_texts.append(" ".join(butter_text))
    return transformed_texts


class ButterFingersTransformation(SentenceOperation):
    """
    Add typos to text the sentence using keyboard distance

    Attributes
    ----------
    args: TransformArguments
        parameters of the transformation
    seed: int
        seed to freeze everything (default is 42)
    max_outputs: int
        maximum number of the transfromed sentences (default is 1)
    device: str
        ! exists for compatability, always ignored !
        the device used during transformation (default is 'cpu')
    spacy_model: spacy.language.Language
        spacy model used for tokenization

    Methods
    -------
    generate(sentence, stop_words, prob)
        Transforms the sentence
    """

    def __init__(
        self,
        args: TransformArguments,
        seed: int = 42,
        max_outputs: int = 1,
        device: Optional[str] = None,
        spacy_model: Optional[Language] = None,
    ) -> None:
        """
        Parameters
        ----------
        args: TransformArguments
            parameters of the transformation
        seed: int
            seed to freeze everything (default is 42)
        max_outputs: int
            maximum number of the transfromed sentences (default is 1)
        device: str
            ! exists for compatability, always ignored !
            the device used during transformation (default is 'cpu')
        spacy_model: spacy.language.Language
            spacy model used for tokenization
        """
        if spacy_model is None:
            spacy_model = spacy.load("ru_core_news_sm")

        super().__init__(
            args=args,
            seed=seed,
            max_outputs=max_outputs,
            device=device,
            spacy_model=spacy_model,
        )

    def generate(
        self,
        sentence: str,
        stop_words: Optional[List[Union[int, str]]] = None,
        prob: Optional[float] = None,
    ) -> List[str]:
        """
        Transforms the sentence

        If 'prob' argument is not None, ignores the probability provided in the arguments.

        Parameters
        ----------
        sentence: str
            sentence to transform
        stop_words: List[int], optional
            stop words to ignore during transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        transformed = butter_finger(
            text=sentence,
            spacy_model=self.spacy_model,
            prob=(self.args.probability if not prob else prob),
            seed=self.seed,
            max_outputs=self.max_outputs,
            stop_words=stop_words,
        )
        return transformed
