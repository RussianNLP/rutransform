import random
import spacy
from typing import List, Optional, Union
from spacy.language import Language

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

"""
Adapted from https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/change_char_case
"""


def change_char_case(
    text: str,
    spacy_model: Language,
    prob: float = 0.1,
    seed: int = 42,
    max_outputs: int = 1,
    stop_words: List[str] = None,
) -> List[str]:
    """
    Changes character cases randomly

    Parameters
    ----------
    text: str
        text to transform
    spacy_model: spacy.language.Language
        spacy model used for lemmatization
    prob: float
        probabilty of the transformation (default is 0.1)
    seed: int
        seed to freeze everything (default is 42)
    max_outputs: int
        maximum number of the returned sentences (default is 1)
    stop_words: List[str], optional
        stop words to ignore during transformation (default is None)

    Returns
    -------
    List[str]
        list of transformed sentences
    """
    if stop_words is None:
        stop_words = []

    random.seed(seed)
    results = []
    split_text = [token.text for token in spacy_model(text)]
    for _ in range(max_outputs):
        result = []
        for w, word in enumerate(split_text):
            if word in stop_words:
                new_word = word
            else:
                new_word = ""
                for c in word:
                    if random.uniform(0, 1) < prob:
                        if c.isupper():
                            new_word += c.lower()
                        elif c.islower():
                            new_word += c.upper()
                    else:
                        new_word += c
            result.append(new_word)
        result = " ".join(result)
        results.append(result)
    return results


class ChangeCharCase(SentenceOperation):
    """
    Changes character cases randomly

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
            the device used during transformation (default is None)
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
        stop_words: List[str], optional
            stop_words to ignore during transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        transformed = change_char_case(
            text=sentence,
            spacy_model=self.spacy_model,
            prob=(self.args.probability if not prob else prob),
            seed=self.seed,
            max_outputs=self.max_outputs,
            stop_words=stop_words,
        )
        return transformed
