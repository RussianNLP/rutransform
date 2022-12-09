import itertools
import spacy
from typing import List, Tuple, Optional, Union

import numpy as np
from spacy.language import Language
from transformers import MarianTokenizer, MarianMTModel

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation


"""
Adapted from https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/back_translation_ner
"""


class BackTranslationNER(SentenceOperation):
    """
    Generates diverse linguistic variations of the contexts
    around the entity mention(s) through back-translation
    ru -> en -> ru using Heksinki-NLP/opus-mt models

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
    spacy_tagger(text, stop_words)
        Tokenizes the sentence and extract entity mentions
    translation_pipeline(text)
        Passes the text in source languages through the intermediate
        translations
    create_segments(tokens, tags)
        Creates segments for translation
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
        if spacy_model is None:
            spacy_model = spacy.load("ru_core_news_sm")

        super().__init__(
            args=args,
            seed=seed,
            max_outputs=max_outputs,
            device=device,
            spacy_model=spacy_model,
        )

        np.random.seed(self.seed)
        mname_ru2en = "Helsinki-NLP/opus-mt-ru-en"
        mname_en2ru = "Helsinki-NLP/opus-mt-en-ru"
        self.tokenizer_ru2en = MarianTokenizer.from_pretrained(mname_ru2en)
        self.tokenizer_en2ru = MarianTokenizer.from_pretrained(mname_en2ru)
        self.model_ru2en = MarianMTModel.from_pretrained(mname_ru2en).to(self.device)
        self.model_en2ru = MarianMTModel.from_pretrained(mname_en2ru).to(self.device)
        self.spacy_model = spacy_model

    def spacy_tagger(
        self, text: str, stop_words: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Tokenizes the sentence and extract entity mentions

        Parameters
        ----------
        text: str
            text to tokenize
        stop_words: List[int], optional
            stop_words to ignore during transformation (default is None)

        Returns
        -------
        Tuple[List[str], List[str]]
            tokenized text, BIO-annotated text
        """
        doc = self.spacy_model(text)
        ner = []
        tokenized = []
        for t, token in enumerate(doc):
            tokenized.append(token.text)
            if token.ent_type_:
                ner.append(token.ent_type_)
            elif stop_words is not None and t in stop_words:
                ner.append("B")
            else:
                ner.append("O")
        return tokenized, ner

    def translation_pipeline(self, text: str) -> str:
        """
        Passes the text in source languages through the intermediate
        translations

        Parameters
        ----------
        text: str
            text to translate

        Returns
        -------
        str
            back-translated text
        """
        ru2en_inputids = self.tokenizer_ru2en.encode(text, return_tensors="pt")
        ru2en_inputids = ru2en_inputids.to(self.device)
        outputs_ru2en = self.model_ru2en.generate(ru2en_inputids)
        text_trans = self.tokenizer_ru2en.decode(
            outputs_ru2en[0], skip_special_tokens=True
        )
        en2ru_inputids = self.tokenizer_en2ru.encode(text_trans, return_tensors="pt")
        en2ru_inputids = en2ru_inputids.to(self.device)
        outputs_en2ru = self.model_en2ru.generate(en2ru_inputids)
        text_trans = self.tokenizer_en2ru.decode(
            outputs_en2ru[0], skip_special_tokens=True
        )
        return text_trans

    @staticmethod
    def create_segments(
        tokens: List[str], tags: List[str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Creates segments for translation

        A segment is defined as a consecutive sequence of same tag/label

        Parameters
        ----------
        tokens: List[str]
            tokenized text
        tags: List[str]
            BIO-annotated text

        Returns
        -------
        Tuple[List[List[str]], List[List[str]]]
            segments of the text and its BIO-annotation
        """
        segment_tokens, segment_tags = [], []
        tags_idxs = [(i, t) for i, t in enumerate(tags)]
        groups = [
            list(g)
            for _, g in itertools.groupby(tags_idxs, lambda s: s[1].split("-")[-1])
        ]
        for group in groups:
            idxs = [i[0] for i in group]
            segment_tokens.append([tokens[idx] for idx in idxs])
            segment_tags.append([tags[idx] for idx in idxs])

        return segment_tokens, segment_tags

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
            stop_words to ignore during transformation (default is None)
        prob: float, optional
            ! exists for compatability, always ignored !
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """

        # tag sentence to extract entity mentions
        token_sequence, tag_sequence = self.spacy_tagger(sentence, stop_words)

        assert len(token_sequence) == len(
            tag_sequence
        ), f"token_sequence and tag_sequence should have same length! {len(token_sequence)}!={len(tag_sequence)}"

        transformations = []
        segment_tokens, segment_tags = BackTranslationNER.create_segments(
            token_sequence, tag_sequence
        )
        for _ in range(self.max_outputs):
            tokens = []
            for s_token, s_tag in zip(segment_tokens, segment_tags):
                if len(s_token) >= 100:
                    segment_text = " ".join(s_token)
                    tokens.extend([segment_text])
                    continue
                translate_segment = np.random.binomial(1, p=self.args.bin_p)
                if (
                    s_tag[0] != "O"
                    or len(s_token) < self.args.segment_length
                    or not translate_segment
                ):
                    tokens.extend(s_token)
                    continue
                segment_text = " ".join(s_token)
                segment_translation = self.translation_pipeline(segment_text)
                tokens.extend([segment_translation])

            transformations.append(" ".join(tokens))

        return transformations
