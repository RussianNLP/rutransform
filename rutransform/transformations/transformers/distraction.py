import random
import re
from typing import Any, Dict, List, Optional, Tuple
from razdel import sentenize

from rutransform.transformations.transformations.distraction import *
from rutransform.transformations import Transformer
from rutransform.constraints import Constraint
from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


class AdditionTransformer(Transformer):
    """
    Addition transformations

    Generates additional sentence or words at the end of the sentence

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
    transform_info:
        dictionary mapping transformations and SentenceOperation classes
        provided in utils.constants
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
    _transform_multichoice(transformer, sentence, add_split)
        Generates new answer options for multichoice questions and
        filters the transformed texts
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
        device: str = str,
        constraints=Optional[List[Constraint]],
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
            transformation_type="addition",
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
            "addsent": SentenceAdditions,
        }

        return info

    def _transform_multichoice(
        self,
        transformer: SentenceOperation,
        sentence: Dict[str, Any],
        add_split: bool = False,
    ) -> Tuple[List[str], List[float]]:
        """
        Generates new answer options for multichoice questions and
        filters the transformed texts

        Parameters
        ----------
        transformer: SentenceOperation
            transformer used for transformation
        sentences: dict
            dataset object in dict form
        add_split: bool
            whether to do additional splitting of the
            generated data (default is False)
            used to trim the generated text to create
            shorter sequences

        Returns
        -------
        Tuple[List[str], List[float]]
            list of transformed texts and their similarity scores
        """
        # split text into context and answer options
        sentences = self.sent_split(sentence[self.text_col])
        context = sentences.pop(0)
        # get answer index
        keys = ["A", "B", "C", "D"]
        answer = sentence[self.label_col]
        answer = keys.index(answer) if type(answer) == str else answer

        transform_sent = [[context]]
        imediate_context = [sentence.text for sentence in sentenize(context)][-1]

        # generate new answers
        change_answ = random.choice(range(len(sentences)))
        while change_answ == answer:
            change_answ = random.choice(range(len(sentences)))

        for s_id, sent in enumerate(sentences):
            if s_id == change_answ:
                transformed = set(transformer.generate(imediate_context))
                if add_split == "sent":
                    transf_sent = []
                    for s in transformed:
                        s = s.replace(imediate_context, "").replace("\n", " ")
                        split_sent = re.split(r"[\.\?!]", s)[0]
                        if len(split_sent) > 1 and len(split_sent[0]) > 1:
                            transf_sent.append(split_sent.strip())
                        else:
                            transf_sent.append(" ".join(s.split()[:5]).strip())
                else:
                    transf_sent = [
                        re.split(
                            r"[\.\?!]",
                            s.replace(imediate_context, "").replace("\n", " "),
                        )[0].strip()
                        for s in transformed
                    ]
                transf_sent = self._drop_duplicates(sent, transf_sent)
                transform_sent.append(transf_sent)
            else:
                transform_sent.append([sent])
        transform_sent = self._sample_to_max_outputs(transform_sent)
        transform_sent, sent_scores = self._filter_candidates(
            transform_sent, sentence[self.text_col]
        )

        return transform_sent, sent_scores

    def transform_sentence(
        self,
        transformer: SentenceOperation,
        sentence: str,
        reference: str,
        context: Optional[str] = None,
        add_split: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Applies the transformations to sentence and filters the transformed sentences

        Parameters
        ----------
        transformer: SentenceOperation
            transformer used for transformation
        sentence: str
            sentence to transform
        reference:
            original sentence
        context: str, optional
            full context (default is None)
        add_split: str, optional
            type of additional splitting to do (default is None)
            - if 'word' returns first 5 words of the generated text
            - if 'sent' returns the first generated sentence (sequence to '.')

        Returns
        -------
        Tuple[List[str], List[float]]
            list of transformed sentences and their similarity scores
        """
        transform_sent = transformer.generate(sentence)

        if add_split == "sent":
            transformed = []
            for s in transform_sent:
                split_sent = re.split(
                    r"[\.\?!]", s.replace(sentence, "").replace("\n", " ")
                )
                if len(split_sent[0]) < 10 and len(split_sent) > 1:
                    transformed.append((sentence + " " + split_sent[1]).strip())
                else:
                    transformed.append((sentence + " " + split_sent[0]).strip())
            transform_sent = transformed
        elif add_split == "word5":
            transform_sent = [
                sentence
                + " "
                + " ".join(
                    s.replace(sentence, "").replace("\n", " ").split()[:5]
                ).strip()
                for s in transform_sent
            ]
        elif add_split == "word3":
            transform_sent = [
                sentence
                + " "
                + " ".join(
                    s.replace(sentence, "").replace("\n", " ").split()[:3]
                ).strip()
                for s in transform_sent
            ]
        else:
            transform_sent = [
                sentence
                + " "
                + " ".join(s.replace(sentence, "").replace("\n", " ")[:10]).strip()
                for s in transform_sent
            ]

        transform_sent = self._drop_duplicates(sentence, transform_sent)
        transform_sent, sent_scores = self._filter_candidates(
            transform_sent, reference, context=context
        )
        transform_sent, sent_scores = self._sample_to_max_outputs(
            transform_sent, sent_scores
        )

        return transform_sent, sent_scores

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
            if self.task_type == "multichoice_qa":
                transform_sent, sent_scores = self._transform_multichoice(
                    transformer, sentence, add_split="sent"
                )

                if len(transform_sent) == 1 and sent_scores[0] == 1:
                    transform_sent, sent_scores = self._transform_multichoice(
                        transformer, sentence, add_split="word"
                    )

            else:
                split_text = self.sent_split(sentence[self.text_col])
                if len(split_text) > 1:
                    context, text = map(
                        lambda x: " ".join(x), (split_text[:-2], split_text[-2:])
                    )
                else:
                    text = sentence[self.text_col]
                    context = None
                transform_sent, sent_scores = self.transform_sentence(
                    transformer,
                    text,
                    sentence[self.text_col],
                    context,
                    add_split="sent",
                )
                for split_type in ["word5", "word3", "char"]:
                    if len(transform_sent) == 1 and sent_scores[0] == 1:
                        transform_sent, sent_scores = self.transform_sentence(
                            transformer,
                            text,
                            sentence[self.text_col],
                            context,
                            add_split=split_type,
                        )
                    else:
                        break

            transformed = self._update_data(sentence, transform_sent)
            transformed_data.extend(transformed)
            scores.extend(sent_scores)

        return transformed_data, scores
