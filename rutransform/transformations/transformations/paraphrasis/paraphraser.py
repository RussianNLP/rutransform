import transformers

from typing import List, Optional, Union
from spacy.language import Language

from russian_paraphrasers import GPTParaphraser, Mt5Paraphraser

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

transformers.logging.set_verbosity_error()


class Paraphraser(SentenceOperation):
    """
    Generates diverse linguistic variations of the contexts
    through paraphrasis using a ruGPT or ruMT5 model

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
            ! exists for compatability, always ignored !
            spacy model used for tokenization
        """
        super().__init__(
            args=args,
            seed=seed,
            max_outputs=max_outputs,
            device=device,
            spacy_model=spacy_model,
        )

        if "gpt" in self.args.generator:
            self.paraphraser = GPTParaphraser(
                model_name=self.args.generator, range_cand=True, make_eval=False
            )
        else:
            self.paraphraser = Mt5Paraphraser(
                model_name=self.args.generator, range_cand=True, make_eval=False
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
        stop_words: List[int], optional
            ! exists for compatability, always ignored !
            stop_words to ignore during transformation (default is None)
        prob: float, optional
            ! exists for compatability, always ignored !
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        transformed = self.paraphraser.generate(
            sentence,
            n=self.max_outputs,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            max_length=self.args.max_length,
            repetition_penalty=self.args.repetition_penalty,
            threshold=self.args.threshold,
        )
        best = transformed["results"][0]["best_candidates"]
        if best:
            return best
        else:
            return transformed["results"][0]["predictions"]
