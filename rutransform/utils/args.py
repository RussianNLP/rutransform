import logging

from dataclasses import field, dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class TransformArguments:
    """
    Arguments pertaining to transformations we are going to apply to data
    """

    transformation: Optional[str] = field(
        default=None, metadata={"help": "transformation to use for data augmentation"}
    )
    max_outputs: int = field(
        default=1, metadata={"help": "maximum number of the transformed sentences"}
    )
    probability: float = field(
        default=0.1, metadata={"help": "probability of the transformation"}
    )
    same_prob: bool = field(
        default=True,
        metadata={
            "help": "whether to use the same probability for EDA deletion and swap"
        },
    )
    del_prob: float = field(
        default=0.05,
        metadata={
            "help": "probability of the word deletion for EDA transformation "
            "Needs to be specified if same_prob parameter is True"
        },
    )
    similarity_threshold: float = field(
        default=0.8, metadata={"help": "BERTScore value to filter out candidates"}
    )
    bae_model: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "BERT model for BAE attack"},
    )
    segment_length: int = field(
        default=3, metadata={"help": "minimal segment length for BackTranslationNE"}
    )
    bin_p: float = field(
        default=1.0,
        metadata={
            "help": "parameter of the binomial distribution for BackTranslationNE"
        },
    )
    generator: str = field(
        default="gpt3",
        metadata={
            "help": "generator model: 'gpt2' = sberbank-ai/rugpt2large, "
            "'gpt3' = sberbank-ai/rugpt3small_based_on_gpt2, "
            "'mt5-small' = google/mt5-small, 'mt5-base' =  google/mt5-base, "
            "'mt5-large' = google/mt5-large"
        },
    )
    prompt_text: str = field(
        default=" Парафраза:", metadata={"help": "prompt for text generation"}
    )
    prompt: bool = field(
        default=False, metadata={"help": "whether to use a prompt for generation"}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "number of beams for beam search"}
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "whether to stop when beam hypotheses reached the EOS token"},
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=None, metadata={"help": "n-gram penalty for beam search generation"}
    )
    do_sample: bool = field(default=False, metadata={"help": "whether to do sampling"})
    temperature: Optional[float] = field(
        default=None, metadata={"help": "temperature for text generation"}
    )
    top_k: Optional[int] = field(
        default=None, metadata={"help": "top-k sampling parameter for text generation"}
    )
    top_p: Optional[float] = field(
        default=None, metadata={"help": "top-p sampling parameter for text generation"}
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "repetition penalty parameter for text generation"},
    )
    threshold: Optional[float] = field(
        default=None, metadata={"help": "threshold parameter to filter candidates"}
    )
    max_length: int = field(
        default=50, metadata={"help": "maximum length of the generated text"}
    )

    def __post_init__(self):
        if self.transformation is None:
            logger.warning("No transformation was passed.")
