from spacy.language import Language

from typing import Optional, List, Dict, Any

from rutransform.constraints import Constraint


class Multihop(Constraint):
    """
    Constraints for multihop QA tasks

    Matches all the bridge and main answers important
    for hops

    Attributes
    ----------
    name: str
        name of the constraint (is always 'referents')
    bridge_col_name: str
        name of the column containing bridge answers
    main_col_name: str
        name of the column containig main question answers

    Methods
    -------
    extract_words(answer)
        Parses answer dictionary and extracts all tokens
    patterns(text, spacy_model)
        Creates spacy.Matcher patterns to extract stopwords
    """

    def __init__(self, bridge_answers_col: str, main_answers_col: str) -> None:
        """
        Parameters
        ----------
        bridge_col_name: str
            name of the column containing bridge answers
        main_col_name: str
            name of the column containig main question answers
        """
        super().__init__(name="multihop")
        self.bridge_answers_col = bridge_answers_col
        self.main_answers_col = main_answers_col

    def extract_words(self, answers: Dict[str, Any]) -> List[str]:
        """
        Parses answer dictionary and extracts all tokens

        Parameters
        ----------
        answers: Dict[str, Any]
            answers dictionary

        Returns
        -------
        List[str]
            list of tokens in the answer
        """
        stop_words = []
        for answer in answers:
            stop_words.extend(answer["segment"].split())
        return stop_words

    def patterns(
        self, text: Optional[dict] = None, spacy_model: Optional[Language] = None
    ) -> List[List[dict]]:
        """
        Creates spacy.Matcher patterns to extract stopwords

        Parameters
        ----------
        text: dict
            dataset object in dict form (default is None)
        spacy_model: spacy.language.Language
            spacy model to be used for morphological analysis (default is None)

        Returns
        -------
        List[List[dicMult]]
            list of spacy.Matcher patterns matching entities important for hops
        """
        stop_words = self.extract_words(text[self.bridge_answers_col])
        stop_words += self.extract_words(text[self.main_answers_col])

        stop_words = list(set(stop_words))
        patterns = [[{"TEXT": {"IN": stop_words}}]]

        return patterns
