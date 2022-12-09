from typing import List, Optional
from spacy.language import Language

from rutransform.constraints import Constraint


class Jeopardy(Constraint):
    """
    Jeopardy type conatraints, including:
    - Noun Phrases such as THIS FILM, THIS ACTOR, both UPPER and lower cased
    - 'X'
    - «Named Entity in parentheses»

    Attributes
    ----------
    name: str
        name of the constraint (is always 'jeopardy')
    lemmas: List[str], optional
        lemmas to include in the patterns (default is None)
        used to define the list of DET that can be used in
        jeopardy questions (e.g. if we want to include 'this' but not 'that')

    Methods
    -------
    patterns(text, spacy_model)
        Creates spacy.Matcher patterns to extract stopwords
    """

    def __init__(self, lemmas: Optional[List[str]] = None) -> None:
        """
        Parameters
        ----------
        lemmas: List[str]
            lemmas to include in the patterns (default is None)
            used to define the list of DET that can be used in
            jeopardy questions (e.g. if we want to include 'this' but not 'that')
        """
        super().__init__(name="jeopardy")
        self.lemmas = lemmas

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
            spacy model to be used for morphological analysis (default is None)

        Returns
        -------
        List[List[dict]]
            list of spacy.Matcher patterns matching jeopardy questions
        """

        patterns = [
            [
                {
                    "IS_UPPER": True,
                    "OP": "+",
                    "POS": {"IN": ["NOUN", "PROPN", "DET", "PRON"]},
                }
            ],
            [
                {"IS_UPPER": True, "POS": {"NOT_IN": ["ADP"]}},
                {"POS": "ADJ", "OP": "*"},
                {"POS": "NOUN", "OP": "+"},
            ],
            [
                {"TEXT": "«"},
                {"IS_TITLE": True},
                {"TEXT": {"REGEX": "\w|\d|['?!.]"}, "OP": "*"},
                {"TEXT": "»"},
            ],
        ]

        if self.lemmas is None:
            self.lemmas = [
                "его",
                "ему",
                "её",
                "икс",
                "ими",
                "их",
                "него",
                "ней",
                "неё",
                "ним",
                "них",
                "нём",
                "он",
                "она",
                "они",
                "оно",
                "такой",
                "это",
                "этот",
            ]
        patterns.append(
            [
                {"LEMMA": {"IN": self.lemmas}},
                {"POS": "ADJ", "OP": "*"},
                {"POS": "NOUN", "OP": "+"},
            ]
        )

        return patterns
