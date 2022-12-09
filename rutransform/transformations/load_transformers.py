import os
import pkgutil
from pathlib import Path
from importlib import import_module
import inspect
from rutransform.transformations import Transformer
from rutransform.transformations.utils import SentenceOperation


def load_transformers():
    search = "transformers"
    package_dir = Path(__file__).resolve()
    transformations_dir = package_dir.parent.joinpath(search)
    a = pkgutil.iter_modules(path=[transformations_dir])

    transform_dict = {}
    for (_, folder, _) in a:

        t = import_module(f"rutransform.transformations.transformers.{folder}")

        for name, obj in inspect.getmembers(t):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Transformer)
                and not issubclass(obj, SentenceOperation)
            ):
                try:
                    info = obj.transform_info().items()
                    for transformation, _ in info:
                        transform_dict[transformation] = obj
                except NotImplementedError:
                    pass

    return transform_dict
