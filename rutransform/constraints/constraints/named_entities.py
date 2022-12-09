from typing import List, Optional
from spacy.language import Language

from rutransform.constraints import Constraint


class NamedEntities(Constraint):
    """
    Named entities constraint

    Matches all the named entities in text

    Attributes
    ----------
    name: str
        name of the constraint (is always 'named_entities')
    entity_types: List[str], optional
        list of named entity types to include (default is None)
        matches all types if not provided

    Methods
    -------
    patterns(text, spacy_model)
        Creates spacy.Matcher patterns to extract stopwords
    """

    def __init__(self, entity_types: Optional[List[str]] = None) -> None:
        """
        Parameters
        ----------
        entity_types: List[str], optional
            list of named entity types to include (default is None)
            matches all types if not provided
        """
        super().__init__(name="named_entities")
        self.entity_types = entity_types

    def patterns(
        self, text: Optional[dict] = None, spacy_model: Optional[Language] = None
    ) -> List[List[dict]]:
        """
        Creates spacy.Matcher patterns to extract stopwords

        Parameters
        ----------
        text: dict
            ! exists for compatability, always ignored !
            dataset object in dict form (default is None)
        spacy_model: spacy.language.Language
            ! exists for compatability, always ignored !
            spacy model to be uses for morphological analysis (default is None)

        Returns
        -------
        List[List[dict]]
            list of spacy.Matcher patterns matching named entities
        """
        if self.entity_types is None:
            patterns = [[{"ENT_TYPE": "", "OP": "!"}]]
        else:
            patterns = [[{"ENT_TYPE": {"IN": self.entity_types}, "OP": "!"}]]
        return patterns
