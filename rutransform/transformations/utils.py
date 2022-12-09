from typing import List, Optional, Union, NamedTuple

import pandas as pd
import numpy as np
from datasets import Dataset
from spacy.language import Language
from rutransform.utils.args import TransformArguments


class TransformResult(NamedTuple):
    transformed_dataset: Union[pd.DataFrame, Dataset]
    scores: np.array
    score: float
    std: float


class SentenceOperation(object):
    """
    Generic operation class.

    Attributes
    ----------
    args: TransformArguments
        parameters of the transformation
    seed: int
        seed to freeze everything (default is 42)
    max_outputs: int
        maximum number of the transfromed sentences (default is 1)
    device: str
        the device used during transformation (default is 'cpu')
    spacy_model: spacy.language.Language
        spacy model used for tokenization

    Methods
    -------
    generate(sentence, stop_words)
        Transforms the sentence
    """

    def __init__(
        self,
        args: TransformArguments,
        seed: int = 42,
        max_outputs: int = 1,
        device: str = "cpu",
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
            the device used during transformation (default is 'cpu')
        spacy_model: spacy.language.Language
            spacy model used for tokenization
        """
        self.args = args
        self.seed = seed
        self.max_outputs = max_outputs
        self.device = device
        self.spacy_model = spacy_model

    def generate(
        self,
        sentence: str,
        stop_words: Optional[List[Union[int, str]]] = None,
        prob: Optional[float] = None,
    ) -> List[str]:
        """
        Transforms the sentence

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
        raise NotImplementedError
