import os
import random
import spacy
from json import load
from typing import Dict, List, Optional, Union

from spacy.language import Language

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

"""
Adapted from https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/emojify
"""


def emojify(
    sentence: str,
    word_to_emoji: Dict[str, str],
    spacy_model: Language,
    prob: float = 0.1,
    seed: int = 0,
    max_outputs: int = 1,
    stop_words: Optional[List[str]] = None,
) -> List[str]:
    """
    Randomly replaces tokens with corresponding emojis

    Parameters
    ----------
    sentence: str
        sentence to transform
    word_to_emoji: Dict[str, str]
        dictionary with emojis and their meanings
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
    random.seed(seed)
    doc = spacy_model(sentence)
    results = []

    if stop_words is None:
        stop_words = []

    for _ in range(max_outputs):

        # Reconstruct the sentence with replaced lemma
        transformed_sentence = ""

        for t, token in enumerate(doc):
            lemma = token.lemma_.lower()

            if t in stop_words:
                transformed_sentence += token.text_with_ws

            # Handle numeric tokens
            elif lemma.isnumeric():
                if random.uniform(0, 1) < prob:
                    for digit in list(lemma):
                        emoji = digit
                        if digit in word_to_emoji:
                            emoji = random.choice(word_to_emoji[digit])
                        transformed_sentence += emoji

                    if " " in token.text_with_ws:
                        transformed_sentence += " "

                else:
                    transformed_sentence += token.text_with_ws

            elif lemma in word_to_emoji:
                # We have `prob` chance to replace this token with emoji
                if random.uniform(0, 1) < prob:

                    # Randomly choose a emoji candidate for this token
                    emoji = random.choice(word_to_emoji[lemma])
                    transformed_sentence += emoji

                    if " " in token.text_with_ws:
                        transformed_sentence += " "

                else:
                    transformed_sentence += token.text_with_ws

            else:
                # If lemma is not in the emoji dictionary, we keep it the same
                transformed_sentence += token.text_with_ws

        results.append(transformed_sentence)

    return results


class EmojifyTransformation(SentenceOperation):
    """
    Augments the input sentence by swapping words
    into emojis with similar meanings

    Attributes
    ----------
    args: TransformArguments
        parameters of the transformation
    spacy_model: spacy.language.Language
        spacy model used for tokenization
    seed: int
        seed to freeze everything (default is 42)
    max_outputs: int
        maximum number of the transfromed sentences (default is 1)
    device: str
        ! exists for compatability, always ignored !
        the device used during transformation (default is 'cpu')

    Methods
    -------
    generate(sentence, stop_words, prob)
        Transforms the sentence
    """

    def __init__(
        self,
        args: TransformArguments,
        spacy_model: Optional[Language] = None,
        seed: int = 42,
        max_outputs: int = 1,
        device: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        args: TransformArguments
            parameters of the transformation
        spacy_model: spacy.language.Language
            spacy model used for tokenization
        seed: int
            seed to freeze everything (default is 42)
        max_outputs: int
            maximum number of the transfromed sentences (default is 1)
        device: str
            ! exists for compatability, always ignored !
            the device used during transformation (default is None)
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

        emoji_dict_path = "emoji_dict_ru.json"
        # Load the emoji dictionary
        dict_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), emoji_dict_path
        )
        self.word_to_emoji = load(open(dict_path, "r"))

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
        stop_words: List[Union[int, str]], optional
            stop_words to ignore during transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        transformed = emojify(
            sentence=sentence,
            word_to_emoji=self.word_to_emoji,
            spacy_model=self.spacy_model,
            prob=(self.args.probability if not prob else prob),
            seed=self.seed,
            max_outputs=self.max_outputs,
            stop_words=stop_words,
        )
        return transformed
