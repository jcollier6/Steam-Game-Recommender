"""ML pipeline package exposing the individual processing chunks.

The modules are imported eagerly with the exception of ``train_pipeline`` which
is loaded lazily to avoid ``runpy`` warnings when executing
``python -m ml.pipeline.train_pipeline``.
"""

from importlib import import_module

from . import utils
from . import chunk1_loader
from . import chunk2_structured
from . import chunk3_tags_pca
from . import chunk4_text_pca
from . import chunk5_item_meta
from . import chunk6_faiss
from . import chunk7_user_profile
from . import chunk8_embeddings
from . import chunk9_model
from . import chunk10_train
from . import chunk11_inference

__all__ = [
    "utils",
    "chunk1_loader",
    "chunk2_structured",
    "chunk3_tags_pca",
    "chunk4_text_pca",
    "chunk5_item_meta",
    "chunk6_faiss",
    "chunk7_user_profile",
    "chunk8_embeddings",
    "chunk9_model",
    "chunk10_train",
    "chunk11_inference",
    "train_pipeline",
]


def __getattr__(name: str):
    if name == "train_pipeline":
        module = import_module(".train_pipeline", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

