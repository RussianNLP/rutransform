import random

from typing import Any, Dict, List, Optional

from razdel import sentenize

from rutransform.transformations.transformations.paraphrasis import *
from rutransform.transformations import Transformer
from rutransform.constraints import Constraint
from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


class ParaphrasisTransformer(Transformer):
    """
    Paraphrasis transformations

    Generates diverse linguistic variations of the contexts
    through paraphrasis, back-translation and style-trasfer

    Utilizes constraints provided by the user to extract stopwords
    specific for the task, to which the transformations do not apply.
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
            transformation_type="paraphrasis",
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
            "paraphraser": Paraphraser,
            "style_transfer": None,
            "back_translation": BackTranslationNER,
        }
        return info

    def _transform_text(
        self,
        transformer: SentenceOperation,
        sentences: List[str],
        reference: str,
        stop_words: Optional[List[Union[str, int, List[Union[str, int]]]]] = None,
        prob: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Applies the transformations to long text and filters the transformed texts

        Parameters
        ----------
        transformer: SentenceOperation
            transformer used for transformation
        sentences: List[str]
            sentences of the text to transform
        reference: str
            original sentence to use as a reference for similarity score
        stop_words: List[Union[str, int]], optional
            stop_words for the transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)
            used when no transormed sentence passes the similarity
            score threshold

        Returns
        -------
        Tuple[List[str], List[float]]
            list of transformed texts and their similarity scores
        """
        transform_text = []
        text_scores = []
        for s, sent in enumerate(sentences):

            if stop_words is not None:
                stops = stop_words[s]
            else:
                stops = None

            if s == 0 and self.task_type == "multichoice":
                question_text = []
                sents = [sentence.text for sentence in sentenize(sent)]

                for i, sentence in enumerate(sents):
                    if random.uniform(0, 1) >= self.args.probability:
                        question_text.append([sentence] * self.args.max_outputs)
                    else:
                        transform_sent = transformer.generate(
                            sentence=sentence, stop_words=stops, prob=prob
                        )
                        question_text.append(
                            transform_sent
                        )  # else random.choice(sentence)

                transformed = []
                for m in range(self.args.max_outputs):
                    transf_sentence = []
                    for tr_sent in question_text:
                        if len(tr_sent) >= m:
                            transf_sentence.append(tr_sent[m])
                        else:
                            transf_sentence.append(random.choice(tr_sent))
                    transformed.append(". ".join(transf_sentence))

                transform_sent = transformed
                transform_sent = self._drop_duplicates(sent, transform_sent)
                transform_sent, sent_scores = self._filter_candidates(
                    candidates=transform_sent, reference=sent
                )
                transform_text.append(transform_sent)
                text_scores.append(sent_scores)

            elif random.uniform(0, 1) >= self.args.probability:
                transform_text.append([sent])
                text_scores.append([1.0])

            else:
                transform_sent = transformer.generate(
                    sentence=sent, stop_words=stops, prob=prob
                )
                transform_sent = self._drop_duplicates(sent, transform_sent)
                transform_sent, sent_scores = self._filter_candidates(
                    candidates=transform_sent, reference=sent
                )
                transform_text.append(transform_sent)
                text_scores.append(sent_scores)

        transform_sent, scores = self._sample_to_max_outputs(
            transform_text, text_scores
        )
        transform_sent, scores = self._filter_candidates(
            candidates=transform_sent, reference=reference
        )
        transform_sent, scores = self._drop_duplicates(
            reference, transform_sent, scores
        )
        return transform_sent, scores

    def transform(self, sentence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applies the transformations to input

        Parameters
        ----------
        sentence: Dict[str, Any]
            dataset object in dict form

        Returns
        -------
        Dict[str, Any]
            Transformed dataset object
        """
        transformed_data = []
        scores = []
        for transform_name, transformer in self.transformers.items():
            stop_words = self._list_stop_words(sentence)
            sentences = self.sent_split(sentence[self.text_col])
            if len(sentences) > 1:
                transform_sent, sent_scores = self._transform_text(
                    transformer,
                    sentences,
                    reference=sentence[self.text_col],
                    stop_words=stop_words,
                )
                if len(transform_sent) == 1 and sent_scores[0] == 1:
                    transform_sent, sent_scores = self._transform_text(
                        transformer,
                        sentences,
                        reference=sentence[self.text_col],
                        stop_words=stop_words,
                        prob=self.args.probability,
                    )
            else:
                transform_sent, sent_scores = self._transform_sentence(
                    transformer, sentence[self.text_col], stop_words=stop_words
                )
            transformed = self._update_data(sentence, transform_sent)
            transformed_data.extend(transformed)
            scores.extend(sent_scores)

        return transformed_data, scores
