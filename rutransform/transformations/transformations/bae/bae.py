from typing import List, Optional, Union
from spacy.language import Language

from textattack.augmentation import Augmenter as TAAugmenter
from textattack.transformations import WordSwapMaskedLM
from textattack.constraints.pre_transformation.stopword_modification import (
    StopwordModification,
)

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


class BAE(SentenceOperation):
    """
    BERT masked language model transformation attack from
    "BAE: BERT-based Adversarial Examples for Text Classification"
    (Garg & Ramakrishnan, 2019).

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
        ! exists for compatability, always ignored !
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
        super().__init__(
            args=args,
            seed=seed,
            max_outputs=max_outputs,
            device=device,
            spacy_model=spacy_model,
        )

        self.transformation = WordSwapMaskedLM(
            method="bae",
            masked_language_model=self.args.bae_model,
            tokenizer=self.args.bae_model,
        )

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
            ! exists for compatability, always ignored !
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        if stop_words is not None:
            constraints = [StopwordModification(stop_words)]

            augmenter = TAAugmenter(
                transformation=self.transformation,
                transformations_per_example=self.max_outputs,
                constraints=constraints,
            )
        else:
            augmenter = TAAugmenter(
                transformation=self.transformation,
                transformations_per_example=self.max_outputs,
            )

        perturbed = augmenter.augment(sentence)

        return perturbed
