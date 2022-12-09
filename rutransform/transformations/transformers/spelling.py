from typing import Any, Dict, List, Optional, Tuple

from rutransform.transformations.transformations.spelling import *
from rutransform.transformations import Transformer
from rutransform.constraints import Constraint
from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


class SpellingTransformer(Transformer):
    """
    Spelling transformations

    Adds noise to text by mimicking common spelling mistakes,
    such as case alteration and keyboard distance character swaps,
    and emoji replacement

    Uses similarity metric (BERTScore) to filter the sentences,
    similarity score of which are less than a threshold (specified in
    TransformArguments).

    Attributes
    ----------
    transformation_type: str
        type of the transformations supported by the transformer
    transformations: List[str]
        list of transformations to apply to data
    task_type: str
        type of the task (e.g. 'classification', 'multichoice_qa', etc.)
    args: TransformArguments
        parameters of the transformation
    text_col: str, optional
        name of the column containing text to transform (default is 'text')
    label_col: str, optional
        name of the target column (default is 'label')
    seed: int
        seed to freeze everything (default is 42)
    device: str
        device used during transformation (default is 'cpu')
    constraints: List[Constraint], optional
        list of transformation constraints (default is None)
    spacy_model: spacy.language.Language
        spacy model used for tokenization (default is 'ru_core_news_sm')
    bert_scorer: BERTScorer
        similarity metric  class used to filter transformed texts (default is None)

    Methods
    -------
    @staticmethod
    transform_info: Dict[str, SentenceOperation]
        dictionary mapping transformations and SentenceOperation classes
    load_transformations()
        Loads all the transformations required
    @abstractmethod
    transform(sentence)
        Applies the transformations to input
    sent_split(text)
        Splits text into sentences
    @staticmethod
    get_ids(matches)
        Returns ids of stopwords
    _transform_text(transformer, sentences, reference, stop_words, prob)
        Applies the transformations to long text and filters the transformed texts
    _transform_sentence(self, transformer, sentence, stop_words, prob)
        Applies the transformations to sentence and filters the transformed sentences
    _apply_transformation(transformer, sentence, sentences, reference, stop_words, prob)
        Applies the transformations to text until the transformed text passes
        the similarity threshold
    _list_stop_words(sentence, return_ids)
        Extracts stopwords matching the constraints
    _drop_duplicates(reference, candidates, scores)
        Returns transformed sentences without duplicates
    _filter_candidates(candidates, reference, context)
        Filters out sentences based on the similarity score
    _sample_to_max_outputs(sentences, scores)
        Returns the desired number of the transformed sentences
    _update_data(org_sentence, transformed)
        Updates the dataset object
    """

    def __init__(
        self,
        transformations: List[str],
        task_type: str,
        args: TransformArguments,
        text_col: Optional[str] = "text",
        label_col: Optional[str] = "label",
        seed: int = 42,
        device: str = "cpu",
        constraints: Optional[List[Constraint]] = None,
    ) -> None:
        """
        Parameters
        ----------
        transformations: List[str]
            list of transformations to apply to data
        task_type: str
            type of the task (e.g. 'classification', 'multichoice_qa', etc.)
        args: TransformArguments
            parameters of the transformation
        text_col: str, optional
            name of the column containing text to transform (default is 'text')
        label_col: str, optional
            name of the target column (default is 'label')
        seed: int
            seed to freeze everything (default is 42)
        device: str
            device used during transformation (default is 'cpu')
        constraints: List[Constraint], optional
            list of transformation constraints (default is None)
        """
        super().__init__(
            transformation_type="spelling",
            transformations=transformations,
            task_type=task_type,
            args=args,
            text_col=text_col,
            label_col=label_col,
            seed=seed,
            device=device,
            constraints=constraints,
        )

        self.transformers = self.load_transformations()

    @staticmethod
    def transform_info() -> Dict[str, Optional[SentenceOperation]]:
        """
        Information about the transformations used by the transformer

        Returns
        -------
        Dict[str, Optional[SentenceOperation]]
            dictionary storing transformation info
        """
        info = {
            "butter_fingers": ButterFingersTransformation,
            "case": ChangeCharCase,
            "emojify": EmojifyTransformation,
        }

        return info

    def _apply_transformation(
        self,
        transformer: SentenceOperation,
        sentence: str,
        sentences: Optional[List[str]] = None,
        stop_words: Optional[List[Union[str, int, List[Union[str, int]]]]] = None,
        prob: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Applies the transformations to text until the transformed text passes
        the similarity threshold

        Parameters
        ----------
        transformer: SentenceOperation
            transformer used for transformation
        sentence: str
            original sentence
        sentences: List[str], optional
            list of sentences of the text to transform if working
            with long texts (default is None)
        stop_words: List[Union[str, int]], optional
            stop_words for the transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)

        Returns
        -------
        Tuple[List[str], List[float]]
            list of transformed texts and their similarity scores
        """

        if sentences is not None:
            transform_sent, sent_scores = self._transform_text(
                transformer, sentences, sentence, stop_words=stop_words, prob=prob
            )
            prob = prob / 2
            count = 0
            while prob > 0.01 and count < 3:
                if len(transform_sent) == 1 and transform_sent[0] == sentence:
                    transform_sent, sent_scores = self._transform_text(
                        transformer, sentences, sentence, prob=prob
                    )
                    count += 1
                    prob = prob / 2
                else:
                    break
            return transform_sent, sent_scores

        transform_sent, sent_scores = self._transform_sentence(
            transformer, sentence, stop_words=stop_words, prob=prob
        )
        prob = prob / 2
        count = 0
        while prob > 0.01 and count < 3:
            if len(transform_sent) == 1 and transform_sent[0] == sentence:
                transform_sent, sent_scores = self._transform_sentence(
                    transformer, sentence, stop_words=stop_words, prob=prob
                )
                prob = prob / 2
                count += 1
            else:
                break

        return transform_sent, sent_scores

    def transform(self, sentence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applies the transformations to input

        Parameters
        ----------
        sentence: dict
            dataset object in dict form

        Returns
        -------
        dict
            Transformed dataset object
        """
        transformed_data = []
        scores = []
        for transform_name, transformer in self.transformers.items():
            if self.task_type in ["multihop", "multichoice_qa"]:
                stop_words = self._list_stop_words(sentence)
                sentences = self.sent_split(sentence[self.text_col])
                transform_sent, sent_scores = self._apply_transformation(
                    transformer,
                    sentence[self.text_col],
                    sentences=sentences,
                    stop_words=stop_words,
                    prob=self.args.probability,
                )

            else:
                stop_words = self._list_stop_words(sentence)
                transform_sent, sent_scores = self._apply_transformation(
                    transformer,
                    sentence[self.text_col],
                    stop_words=stop_words,
                    prob=self.args.probability,
                )

            transformed = self._update_data(sentence, transform_sent, transform_name)
            transformed_data.extend(transformed)
            scores.extend(sent_scores)

        return transformed_data, scores
