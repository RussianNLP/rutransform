import logging
import random
import re
import spacy
from spacy.matcher import Matcher
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Union, Optional, Tuple, Any

from razdel import sentenize
from bert_score import BERTScorer

from rutransform.constraints import Constraint
from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


class Transformer(object):
    """
    Base Class for implementing the different input transformations

    Takes sentence as input and applies the chosen transformations.
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
    constraints: List[Constraint]
        list of transformation constraints
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
        transformation_type: str,
        transformations: List[str],
        task_type: str,
        args: TransformArguments,
        text_col: Optional[str] = "text",
        label_col: Optional[str] = "label",
        seed: int = 42,
        device: str = "cpu",
        constraints: List[Constraint] = None,
    ) -> None:
        """
        Parameters
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
        constraints: List[Constraint]
            list of transformation constraints (default is None)
        """
        self.transformations = transformations
        self.task_type = task_type
        self.args = args
        self.text_col = text_col
        self.label_col = label_col
        self.seed = seed
        self.device = device
        self.constraints = constraints

        self.spacy_model = spacy.load("ru_core_news_sm")
        self.bert_scorer = BERTScorer(lang="ru")

    @staticmethod
    def transform_info() -> Dict[str, Optional[SentenceOperation]]:
        """
        Information about the transformations used by the transformer

        info = {
            transformation_name: SentenceOperation transformation class
            }

        Returns
        -------
        Dict[str, Optional[SentenceOperation]]
            dictionary storing transformation info
        """
        raise NotImplementedError

    def load_transformations(self) -> Dict[str, SentenceOperation]:
        """
        Load all the transformations required for transformations

        When TransformArguments.max_ouputs number is < 3 and the
        transformation is not computaionally costly sets max_outputs
        parameter of the transformation to 3 to generate more possible
        adversarial examples. This is to ensure more examples are
        left after the filtration.

        Returns
        -------
        Dict[str, SentenceOperation]
            initialized transformations

        Raises
        ------
        ValueError
            if passed transformation does not in the list of possible transformations
        NotImplementedError
            if passed transformations is not yet implemented
        """
        transformers = {}
        for transformation in self.transformations:
            logging.info("Transformation: %s" % transformation)
            if transformation not in self.transform_info():
                raise ValueError("Invalid transformation name: %s" % transformation)
            if not self.transform_info()[transformation]:
                raise NotImplementedError

            compute_cost = False
            if transformation in ["bae", "parapharsis", "back_translation", "addsent"]:
                compute_cost = True

            transformers[transformation] = self.transform_info()[transformation](
                args=self.args,
                seed=self.seed,
                max_outputs=(
                    self.args.max_outputs
                    if self.args.max_outputs >= 3 or compute_cost
                    else 3
                ),
                device=self.device,
                spacy_model=self.spacy_model,
            )
        return transformers

    @abstractmethod
    def transform(self, sentence: dict) -> dict:
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
        raise NotImplementedError

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

    def _transform_sentence(
        self,
        transformer: SentenceOperation,
        sentence: str,
        stop_words: Optional[List[Union[str, int, List[Union[str, int]]]]] = None,
        prob: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Applies the transformations to sentence and filters the transformed sentences

        Parameters
        ----------
        transformer: SentenceOperation
            transformer used for transformation
        sentence: str
            sentence to transform
        stop_words: List[Union[str, int]], optional
            stop_words for the transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)
            used when no transormed sentence passes the similarity
            score threshold

        Returns
        -------
        Tuple[List[str], List[float]]
            list of transformed sentences and their similarity scores
        """
        transform_sent = transformer.generate(
            sentence=sentence, stop_words=stop_words, prob=prob
        )
        transform_sent = self._drop_duplicates(sentence, transform_sent)
        transform_sent, sent_scores = self._filter_candidates(
            candidates=transform_sent, reference=sentence
        )
        transform_sent, sent_scores = self._sample_to_max_outputs(
            transform_sent, sent_scores
        )
        return transform_sent, sent_scores

    def sent_split(self, text: str) -> List[str]:
        """
        Splits text into sentences

        When task_type is 'multichoice_qa' splits sentence into
        question and answer options. In other cases splits text
        into sentences.

        Parameters
        ----------
        text: str
            text to split

        Returns
        -------
        List[str]
            list of sentences
        """
        if self.task_type == "multichoice_qa":
            return list(map(str.strip, re.split("\([ABCD]\)", text)))
        else:
            return [sent.text for sent in sentenize(text)]

    @staticmethod
    def get_ids(matches: Tuple[int, int, int]) -> List[int]:
        """
        Returns ids of stopwords

        Parameters
        ----------
        matches: Tuple[int, int, int]
            Matcher output of (id, start, finish)

        Returns
        -------
        List[int]
            indexes of tokens in the sentence matching the stopwords
        """
        ids = []
        for match in matches:
            ids.extend(list(range(match[1], match[2])))
        return ids

    def _list_stop_words(
        self, sentence: dict, return_ids: bool = True
    ) -> List[Union[str, int]]:
        """
        Extracts stopwords matching the constraints

        Parameters
        ----------
        sentence: dict
            dataset object in dict form
        return_ids: bool
            whether to return indices (default is True)
            if False, returns tokens themselves

        Returns
        -------
        List[Union[str, int]]
            list of stopword indices in sentence or tokens
        """

        if self.constraints is None or len(self.constraints) == 0:
            return None

        stop_words = []
        matcher = Matcher(self.spacy_model.vocab)
        for constraint in self.constraints:
            matcher.add(
                constraint.name, constraint.patterns(sentence, self.spacy_model)
            )

        sentences = self.sent_split(sentence[self.text_col])

        if len(sentences) > 1:
            for sent in sentences:
                doc = self.spacy_model(sent)
                words = matcher(doc)
                words = self.get_ids(words)
                if not return_ids:
                    words = [doc[idx].text for idx in words]
                stop_words.append(words)
        else:
            doc = self.spacy_model(sentence[self.text_col])
            words = matcher(doc)
            words = self.get_ids(words)

            if not return_ids:
                words = [doc[idx] for idx in words]
            stop_words.extend([words])

        if len(stop_words) != 0:
            if type(stop_words[0]) is list and len(sum(stop_words, [])) == 0:
                return None
        else:
            return None

        return stop_words

    def _drop_duplicates(
        self,
        reference: str,
        candidates: List[str],
        scores: Optional[List[float]] = None,
    ) -> Union[Tuple[List[str], List[float]], List[str]]:
        """
        Returns transformed sentences without duplicates

        Parameters
        ----------
        reference: str
            original sentence to compare candidates
        candidates: List[str]
            list of transformed sentences
        scores: List[float], optional
            similarity scores of the passed sentences (default is None)

        Returns
        -------
        Union[Tuple[List[str], List[float]], List[str]]
             list of transformed sentences without duplicates
             and their similarity scores, if passed
        """
        if scores:
            candidates = list(zip(candidates, scores))
            candidates = set(candidates)
            if len(candidates) > 1:
                candidates = [text for text in candidates if text[0] != reference]
            candidates, scores = list(map(list, zip(*candidates)))
            return candidates, scores

        candidates = set(candidates)
        if len(candidates) > 1:
            candidates = [text for text in candidates if text != reference]
        else:
            candidates = list(candidates)
        return candidates

    def _filter_candidates(
        self, candidates: List[str], reference: str, context: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Filters out sentences based on the similarity score

        Uses BERT-score (https://arxiv.org/abs/1904.09675) to
        calculate similarity.

        If no candidates pass the similarity threshold, returns
        the original sentence.

        Paramaters
        ----------
        candidates: List[str]
            list of transformed sentences
        reference: str
            original sentence to compare candidates
        context: str, optional
            full context, used for transformations requiring
            only last couple of sentences (e.g. addsent)
            (default is None)

        Returns
        -------
        Tuple[List[str], List[float]]
            list of filtered sentences and their similarity scores
        """
        filtered = []
        scores = []

        for candidate in candidates:
            if context is not None:
                candidate = context + " " + candidate
            _, _, score = self.bert_scorer.score([candidate], [reference])
            if score >= self.args.similarity_threshold:
                filtered.append(candidate)
                scores.append(score[0].item())

        if len(filtered) == 0:
            filtered += [reference]
            scores.append(1.0)

        top_candidates = sorted(list(zip(filtered, scores)), reverse=True)
        filtered, scores = list(map(list, zip(*top_candidates)))
        return filtered, scores

    def _sample_to_max_outputs(
        self, sentences: List[Union[str, List[str]]], scores: Optional[List[str]] = None
    ) -> Union[Tuple[List[str], List[float]], List[str]]:
        """
        Returns the desired number of the transformed sentences

        For multichoice_qa task return the whole sentence in original form
        (i.e. 'question (A) option (B) option')

        Parameters
        ----------
        sentences: List[Union[str, List[str]]]
            list of transformed sentences
            for multichoice_qa it is a list of lists
            containing question and answer options
        scores: List[float], optional
            similarity scores of the passed sentences (default is None)

        Returns
        -------
        Union[Tuple[List[str], List[float]], List[str]]
             list of transformed sentences without duplicates
             and their similarity scores, if passed
        """
        if type(sentences[0]) is list:
            output = []
            sent_scores = []
            for _ in range(self.args.max_outputs):
                sampled = []
                sampled_scores = []
                for i, sent in enumerate(sentences):
                    idx = random.choice(range(len(sent)))
                    sampled.append(sent[idx])
                    if scores is not None:
                        sampled_scores.append(scores[i][idx])
                if self.task_type == "multichoice_qa":
                    sampled = (
                        sampled[0]
                        + " (A) "
                        + sampled[1]
                        + " (B) "
                        + sampled[2]
                        + " (C) "
                        + sampled[3]
                        + " (D) "
                        + sampled[4]
                    )
                else:
                    sampled = " ".join(sampled)
                output.append(sampled)
                sent_scores.append(sampled_scores)
        else:
            if len(sentences) > self.args.max_outputs:
                output = sentences[: self.args.max_outputs]
                if scores is not None:
                    sent_scores = scores[: self.args.max_outputs]
            else:
                return (sentences, scores) if scores is not None else sentences

        return (output, sent_scores) if scores is not None else output

    def __update_offset(self, answers: Dict[str, Any], text: str) -> Dict[str, Any]:
        updated_answers = []

        answer_counts = Counter([answer["segment"] for answer in answers])
        passed = answer_counts.copy()
        for answer in answers:
            answer_text = answer["segment"]
            if answer_counts[answer_text] > 1:
                passed[answer_text] -= 1
                offsets = text.split(answer_text)
                offset = 0
                for i in range(answer_counts[answer_text] - passed[answer_text]):
                    if i == 0:
                        offset += len(offsets[i])
                    else:
                        offset += len(offsets[i]) + len(answer_text)
            else:
                offset = text.find(answer_text)

            answer["offset"] = offset
            updated_answers.append(answer)
        return updated_answers

    def _update_data(
        self,
        org_sentence: dict,
        transformed: List[str],
        transformation: Optional[str] = None,
    ) -> dict:
        """
        Updates the dataset object

        Parameters
        ----------
        org_sentence: dict
            dataset object in dict form
        transformed: List[str]
            list of transformed sentences
        """
        transformed_data = []
        for sentence in transformed:
            transformed_sent = org_sentence.copy()
            transformed_sent[self.text_col] = sentence

            # update the offsets of answers
            if self.task_type == "multihop" and transformation in [
                "eda",
                "emojify",
                "back_translation",
                "paraphrasis",
            ]:
                transformed_sent[self.label_col] = self.__update_offset(
                    org_sentence[self.label_col], sentence
                )

            transformed_data.append(transformed_sent)
        return transformed_data
