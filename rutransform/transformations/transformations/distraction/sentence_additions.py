import re
from typing import List, Optional, Union
from spacy.language import Language
from transformers import (
    TextGenerationPipeline,
    set_seed,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelWithLMHead,
    MT5Tokenizer,
)

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

"""
Adapted from https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/sentence_additions
"""


def clean(text: str) -> str:
    """
    Cleans text from unwanted characters created
    by the generator model

    Parameters
    ----------
    text: str
        generated text to clean

    Returns
    -------
    str
        clean string of text
    """
    if len(text) > 1:
        text = text.split("===")[0]
        text = " ".join(text.split("\n\n")[:2])
        text = text.replace("\\n", "\n")
        text = text.replace("<UNK>", "")
        text = text.replace("&amp;", "&")
        text = text.replace("lt;", "")
        text = text.replace("gt;", "")
        text = text.split("< EOS>")[0]
        text = text.split("<EOS>")[0]
        text = text.replace("< EOS>", " ")
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<EOS>", " ")
        text = text.replace("< BOS>", " ")
        text = text.replace("<BOS>", " ")
        text = text.replace("< SHORT>", " ")
        text = text.replace("<SHORT>", " ")
        text = text.replace("<LONG>", " ")
        text = text.replace("< LONG>", " ")
        text = text.replace(" ul ", "\n")
        text = text.replace(" pre ", " ")
        text = text.replace(r" /pre ", " ")
        text = text.replace(r" / pre ", " ")
        text = text.replace(r"/code", "\n/code\n")
        text = text.replace(r"/ code", "\n/code\n")
        text = text.replace(" code", "\ncode\n")
        text = text.replace(" hr ", " ")
        text = text.replace(" e f ", "\n")
        text = text.replace("/h1", "\n")
        text = text.replace("nbsp;", " ")
        text = text.replace("/blockquote", "\n")
        text = text.replace(" +", " ")
        text = text.replace("&zwj;", "")
        text = text.replace(".<", ".")
        text = text.replace("/", ".")
        text = text.replace("tml", "")
        text = text.replace("</s", "")
        text = text.replace("..s", "")
        text = text.replace("\xa0", " ")
        text = re.sub("&#[0-9]+;", "", text)
        text = text.replace("ћ", "м").replace("ƒ", "д")
        text = re.sub("\([ABCD]\)", "", text)
        text = text.replace("<extra_id_0>", "")
    return text.strip()


class SentenceAdditions(SentenceOperation):
    """
    Adds generated sentence into provided sentences
    or paragraph to create adversarial examples.

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
    get_model_path()
        Converts model name to model path
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

        model_name = self.get_model_path()
        if "mt5" in self.args.generator:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(
                model_name, pad_token_id=self.tokenizer.eos_token_id
            ).to(self.device)
        elif "t5" in self.args.generator:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, pad_token_id=self.tokenizer.eos_token_id
            ).to(self.device)
        else:
            tokenizer = (
                MT5Tokenizer.from_pretrained(model_name)
                if model_name == "THUMT/mGPT"
                else AutoTokenizer.from_pretrained(model_name)
            )
            model = AutoModelWithLMHead.from_pretrained(
                model_name, pad_token_id=tokenizer.eos_token_id
            )
            self.generator = TextGenerationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=(-1 if self.device == "cpu" else 0),
            )

    def get_model_path(self) -> str:
        """
        Converts model name to model path

        Returns
        -------
        str
            path to model in the HuggingFace library
        """
        model_dict = {
            "gpt2": "sberbank-ai/rugpt2_large",
            "gpt3": "sberbank-ai/rugpt3large_based_on_gpt2",
            "mt5-base": "google/mt5-base",
            "mt5-small": "google/mt5-small",
            "mt5-large": "google/mt5-large",
        }
        return (
            model_dict[self.args.generator]
            if self.args.generator in model_dict
            else self.args.generator
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
        if self.max_outputs == 1:
            set_seed(self.seed)

        if self.args.prompt:
            sentence = sentence + self.args.prompt_text

        transformed = []
        for _ in range(self.max_outputs):
            if "t5" in self.args.generator:
                encoding = self.tokenizer.encode_plus(
                    sentence, pad_to_max_length=True, return_tensors="pt"
                )
                input_ids, attention_masks = (
                    encoding["input_ids"].to(self.device),
                    encoding["attention_mask"].to(self.device),
                )

                beam_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    do_sample=self.args.do_sample,
                    max_length=self.args.max_length,
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    early_stopping=self.args.early_stopping,
                    num_return_sequences=1,
                    repetition_penalty=self.args.repetition_penalty,
                )

                for output in beam_outputs:
                    sent = self.tokenizer.decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    transformed.append(sent)
            else:
                outputs = self.generator(
                    sentence,
                    max_length=self.args.max_length,
                    skip_special_tokens=True,
                    num_return_sequences=1,
                    num_beams=self.args.num_beams,
                    early_stopping=self.args.early_stopping,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    temperature=self.args.temperature,
                    do_sample=self.args.do_sample,
                    repetition_penalty=self.args.repetition_penalty,
                )

                for sents_with_additions in outputs:
                    for key, value in sents_with_additions.items():
                        transformed.append(clean(value))
        return transformed
