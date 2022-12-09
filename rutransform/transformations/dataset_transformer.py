from typing import Union, NamedTuple

import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

from rutransform.transformations import Transformer
from rutransform.transformations.utils import *
from rutransform.constraints import *
from rutransform.utils.args import TransformArguments
from rutransform.transformations import load_transformers


class DatasetTransformer(object):
    """
    Class for dataset transformation

    Attributes
    ----------
    dataset: Union[Dataset, pd.DataFrame]
        dataset to transform
    args: TransformArguments
        transformation parameters
    text_col: str, optional
        name of the column containing text to transform (default is 'text')
    label_col: str, optional
        name of the target column (default is 'label')
    task_type: str, optional
        type of the task (default is None)
        if dataset_name in original tasks,
        defaults to task_type of the dataset
    seed: int, optional
        seed to freeze everything (default is 42)
    device: str
        the device used during transformation (default is 'cpu')
    return_type: str
        type of the transformed dataset (default is 'hf')
        - if 'pd' - returns pandas.DataFrame
        - if 'hf' - returns HuggingFace Dataset
    custom_constraints: List[Constraint]
        list of custom constraints for transformation (defaul is None)
        if not provided, uses contrsaints for task_type,
        else uses only custom ones
    transform_dict: dict
        dictionary containing Transformer classes by transformation
        provided in utils.constants
    transform_info:
        dictionary mapping transformations and SentenceOperation classes
        provided in utils.constants

    Methods
    -------
    load_transformer()
        Loads the transformer used for transformation
    transform()
        Transforms dataset

    """

    def __init__(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        args: TransformArguments,
        text_col: str = "text",
        label_col: str = "label",
        task_type: Optional[str] = None,
        seed: int = 42,
        device: str = "cpu",
        return_type: str = "hf",
        custom_constraints: List[Constraint] = None,
        use_constraints: bool = True,
    ) -> None:
        """
        dataset: Union[Dataset, pd.DataFrame]
            dataset to transform
        args: TransformArguments
            transformation parameters
        text_col: str, optional
            name of the column containing text to transform (default is 'text')
        label_col: str, optional
            name of the target column (default is 'label')
        task_type: str, optional
            type of the task
        seed: int, optional
            seed to freeze everything (default is 42)
        device: str
            the device used during transformation (default is 'cpu')
        return_type: str
            type of the transformed dataset (default is 'hf')
            - if 'pd' - returns pandas.DataFrame
            - if 'hf' - returns HuggingFace Dataset
        custom_constraints: List[Constraint]
            list of custom constraints for transformation (defaul is None)
            if not provided, uses contrsaints for task_type,
            else uses only custom ones
        """
        self.dataset = dataset
        self.args = args
        self.text_col = text_col
        self.label_col = label_col
        self.task_type = task_type
        self.seed = seed
        self.device = device
        self.return_type = return_type
        self.custom_constraints = custom_constraints
        self.use_constraints = use_constraints

        self.transform_dict = load_transformers()
        self.transformer = self.load_transformer()

    def load_transformer(self) -> Transformer:
        """
        Loads the transformer used for transformation.
        Initializes task_type and default constraints for task.

        Returns
        -------
        Transformer
            initialized Transformer class
        """
        if self.args.transformation not in self.transform_dict:
            raise ValueError(
                "Invalid transformation name: %s" % self.args.transformation
            )

        if self.use_constraints:
            if self.custom_constraints is not None:
                constraints = self.custom_constraints
            else:
                constraints = []
                if self.task_type == "multichoice_qa":
                    constraints = [NamedEntities()]
                elif self.task_type == "winograd":
                    constraints = [NamedEntities(), Referents()]
                elif self.task_type == "jeopardy":
                    constraints = [NamedEntities(), Jeopardy()]
                elif self.task_type == "multihop":
                    constraints = [
                        NamedEntities(),
                        Multihop("bridge_answers", "main_answers"),
                    ]
        else:
            constraints = None

        transformer = self.transform_dict[self.args.transformation](
            transformations=[self.args.transformation],
            task_type=self.task_type,
            args=self.args,
            text_col=self.text_col,
            label_col=self.label_col,
            seed=self.seed,
            device=self.device,
            constraints=constraints,
        )

        return transformer

    def transform(self) -> TransformResult:
        """
        Transforms dataset

        Applies provided transformations to dataset.
        Uses constraints to ensure the quality of the transformation.

        Returns
        -------
        TransformResult
            result of the transformation, including
            - transformed dataset
                  type provided during initialization (self.return_type)
            - similarity scores of each transformed text (BERT-score)
            - mean similarity score (BERT-score)
            - standard deviation of the similarity scores

        """

        if type(self.dataset) is pd.DataFrame:
            dataset = Dataset.from_pandas(self.dataset)
        else:
            dataset = self.dataset

        transformed = []
        scores = []
        for sent_ind, sentence in tqdm(
            enumerate(dataset), total=len(dataset), desc="Transforming data"
        ):
            transformed_sentence, sent_scores = self.transformer.transform(sentence)
            transformed.extend(transformed_sentence)
            scores.extend(sent_scores)

        transformed = pd.DataFrame(transformed)

        if self.return_type == "hf":
            transformed = Dataset.from_pandas(transformed)

        scores = np.array(scores)

        return TransformResult(
            transformed_dataset=transformed,
            scores=scores,
            score=scores.mean(),
            std=np.std(scores),
        )
