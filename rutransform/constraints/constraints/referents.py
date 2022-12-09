from typing import List, Optional
from spacy.language import Language

from rutransform.constraints import Constraint
from rutransform.constraints.utils import parse_reference


class Referents(Constraint):
    """
    Constraints for coreference resolution tasks

    Matches
    - the anaphoric pronoun
    - all possible antecedents
    - all verbs referring to antecedents and anaphor

    Attributes
    ----------
    name: str
        name of the constraint (is always 'referents')
    reference_col_name: str, optional
        name of the column containing anaphor
        defaults to 'reference' if not provided
    candidates_col_name: str, optional
        name of the column containig possible antecedents
        defaults to 'Options' or 'options' if not provided

    Methods
    -------
    patterns(text, spacy_model)
        Creates spacy.Matcher patterns to extract stopwords
    """

    def __init__(
        self,
        reference_col_name: Optional[str] = None,
        candidates_col_name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        reference_col_name: str, optional
            name of the column containing anaphor
            defaults to 'reference' if not provided
        candidates_col_name: str, optional
            name of the column containig possible antecedents
            defaults to 'options' if not provided
        """
        super().__init__(name="referents")
        self.reference_col_name = reference_col_name
        self.candidates_col_name = candidates_col_name

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
        List[List[dict]]
            list of spacy.Matcher patterns matching antecedents, anaphors and corresponding verbs
        """
        if not self.reference_col_name:
            if "reference" in text:
                self.reference_col_name = "reference"
            else:
                raise ValueError(
                    "Column 'reference' not found in pd.DataFrame columns. "
                    + "Rename the text column or provide 'reference_col_name' argument."
                )
        if not self.candidates_col_name:
            if "options" in text:
                self.candidates_col_name = "options"
            else:
                raise ValueError(
                    "Column 'options' not found in pd.DataFrame columns. "
                    + "Rename the text column or provide 'candidates_col_name' argument."
                )

        options = (
            eval(text[self.candidates_col_name])
            if type(text[self.candidates_col_name]) is str
            else text[self.candidates_col_name]
        )
        morph = parse_reference(text[self.reference_col_name], spacy_model)
        referents = [morph.get("number")] + [morph.get("gender")]
        referents = [referent for referent in referents if referent]
        patterns = [
            [{"TEXT": {"IN": options + text[self.reference_col_name].strip().split()}}],
            [{"POS": "VERB", "MORPH": {"IS_SUPERSET": referents}}],
        ]
        return patterns
