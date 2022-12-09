import random
import spacy
from random import shuffle
from typing import List, Optional, Union
from nltk.corpus import stopwords
from string import punctuation

from spacy.language import Language

from rutransform.utils.args import TransformArguments
from rutransform.transformations.utils import SentenceOperation

STOPWORDS = stopwords.words("russian")

"""
Adapted from https://github.com/jasonwei20/eda_nlp
"""


def tokenize(text: str, spacy_model: Language) -> str:
    """
    Tokenizes text

    Parameters
    ----------
    text: str
        text to tokenize
    spacy_model: spacy.language.Language
        spacy model used for tokenization

    Returns
    -------
    str
        tokenized text
    """
    return " ".join([token.text for token in spacy_model(text)])


def random_deletion(
    words: List[str], p: float, seed: int, stop_words: Optional[List[int]] = None
) -> List[str]:
    """
    Randomly deletes words from the sentence with probability p

    Parameters
    ----------
    words: List[str]
        list of tokens in the sentence
    p: float
        probability of the deletion
    seed: int
        seed to freeze everything
    stop_words: List[int], optional
        stop_words to ignore during deletion (default is None)

    Returns
    -------
    List[str]
        transformed sentence in tokens
    """
    random.seed(seed)
    if stop_words is None:
        stop_words = []

    # if there's only one word, don't delete it
    if len(words) <= 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for idx, word in enumerate(words):
        if idx in stop_words:
            new_words.append(word)
            continue
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    if new_words == words:
        stopwords = [
            i
            for (i, word) in enumerate(words)
            if (word in STOPWORDS and i not in stop_words)
        ]
        if len(stopwords) > 0:
            random_idx = random.choice(stopwords)
            new_words.pop(random_idx)

    return new_words


def random_swap(
    words: List[str], n: int, seed: int, stop_words: Optional[List[int]] = None
) -> List[str]:
    """
    Randomly swaps two words in the sentence n times

    Parameters
    ----------
    words: List[str]
        list of tokens in the sentence
    n: int
        number of swaps
    seed: int
        seed to freeze everything
    stop_words: List[int], optional
        stop_words to ignore during swaps (default is None)

    Returns
    -------
    List[str]
        transformed sentence in tokens
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, seed, stop_words=stop_words)
    return new_words


def swap_word(
    words: List[str], seed: int, stop_words: Optional[List[int]] = None
) -> List[str]:
    """
    Randomly swap two words in the sentence

    Parameters
    ----------
    words: List[str]
        list of tokens in the sentence
    seed: int
        seed to freeze everything
    stop_words: List[int], optional
        stop_words to ignore during swaps (default is None)

    Returns
    -------
    List[str]
        transformed sentence in tokens
    """
    if stop_words is None:
        stop_words = []

    new_words = words.copy()
    random.seed(seed)
    allowed_ids = [i for (i, word) in enumerate(words) if i not in stop_words]
    if len(allowed_ids) >= 2:
        random_idx_1 = random.choice(allowed_ids)  # test
    else:
        return new_words

    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.choice(allowed_ids)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )

    # if we did not swap any of the words swap any articles, pronouns, etc.
    if new_words == words:
        stopwords = [
            i
            for (i, word) in enumerate(new_words)
            if (word in STOPWORDS and i not in stop_words)
        ]
        if len(stopwords) > 1:
            random_idx_1, random_idx_2 = random.sample(stopwords, k=2)
            new_words[random_idx_1], new_words[random_idx_2] = (
                new_words[random_idx_2],
                new_words[random_idx_1],
            )
    return new_words


def eda(
    sentence: str,
    spacy_model: Language,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 1,
    seed: int = 42,
    stop_words: Optional[List[int]] = None,
) -> List[str]:
    """
    Applies Easy Data Augmentations (random deletion and random swaps) to text

    Parameters
    ----------
    sentence: str
        text to transform
    spacy_model: spacy.language.Language
        spacy model used for tokenization
    alpha_rs: float
        probability of word swap (default is 0.1)
    p_rd: float
        probability of word deletion (default is 0.1)
    num_aug: int
        maximum number of the transformed sentences (default is 1)
    seed: int
        seed to freeze everything (default is 42)
    stop_words: List[int], optional
        stop_words to ignore during swaps (default is None)

    Returns
    -------
    List[str]
        list of transformed sentences
    """
    random.seed(seed)
    sentence = tokenize(sentence, spacy_model)
    words = sentence.split()
    words = [word for word in words if word is not ""]
    num_words = len(words)
    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_rs = max(1, int(alpha_rs * num_words))

    # random swap
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs, seed, stop_words=stop_words)
        augmented_sentences.append(" ".join(a_words))

    # random deletion
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd, seed, stop_words=stop_words)
        augmented_sentences.append(" ".join(a_words))

    shuffle(augmented_sentences)

    # trim to the the desired number of augmented sentences
    augmented_sentences = [s for s in augmented_sentences if s != sentence][:num_aug]

    return augmented_sentences


class RandomEDA(SentenceOperation):
    """
    Augment data using Easy Data Augmentation techniques
    (random deletion and random word swaps)

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

    def generate(
        self,
        sentence: str,
        stop_words: Optional[List[Union[int, str]]] = None,
        prob: Optional[float] = None,
    ) -> List[str]:
        """
        Transforms the sentence

        If 'prob' argument is not None, ignores the probabilityprovided in the arguments.

        Parameters
        ----------
        sentence: str
            sentence to transform
        stop_words: List[int], optional
            stop_words to ignore during transformation (default is None)
        prob: float, optional
            probability of the transformation (default is None)

        Returns
        -------
        list
            list of transformed sentences
        """
        if not prob:
            alpha_rs = self.args.probability
            p_rd = self.args.probability if self.args.same_prob else self.args.del_prob
        else:
            alpha_rs = prob
            p_rd = prob if self.args.same_prob else self.args.del_prob / 2

        transformed = eda(
            sentence=sentence,
            alpha_rs=alpha_rs,
            p_rd=p_rd,
            num_aug=self.max_outputs,
            seed=self.seed,
            spacy_model=self.spacy_model,
            stop_words=stop_words,
        )

        return transformed
