from typing import List, Optional
from abc import abstractmethod
from spacy.language import Language


class Constraint:
    """
    Base class for transformation constraints

    Attributes
    ----------
    name: str
        name of the constraint

    Methods
    -------
    @abstractmethod
    patterns(text, spacy_model)
        Creates spacy.Matcher patterns to extract stopwords
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name: str
            name of the constraint
        """
        self.name = name

    @abstractmethod
    def patterns(
        self, text: Optional[dict], spacy_model: Optional[Language]
    ) -> List[List[dict]]:
        """
        Creates spacy.Matcher patterns to extract stopwords

        Parameters
        ----------
        text: dict
            dataset object in dict form
        spacy_model: spacy.language.Language
            spacy model to be uses for morphological analysis

        Returns
        -------
        List[List[dict]]
            list of spacy.Matcher patterns, that match the constraint
        """
        raise NotImplementedError
