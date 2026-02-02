import pickle
import warnings
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import Union
from xgboost import XGBClassifier


# xgboost
class XGBoostHyperparameters(BaseModel):
    n_estimators: int = Field(
        default=100, ge=1, le=1000, description="The number of trees in the forest."
    )
    max_depth: int = Field(
        default=6, ge=1, le=15, description="The maximum depth of the tree."
    )
    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="The learning rate."
    )
    subsample: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The fraction of the training data to use for each tree.",
    )
    colsample_bytree: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The fraction of the features to use for each tree.",
    )
    gamma: float = Field(
        default=0.0, ge=0.0, le=1.0, description="The gamma parameter."
    )
    reg_alpha: float = Field(
        default=0.0, ge=0.0, le=1.0, description="The alpha parameter."
    )
    reg_lambda: float = Field(
        default=1.0, ge=0.0, le=1.0, description="The lambda parameter."
    )


class XGBoostModel:
    def __init__(self, hyperparameters: Optional[XGBoostHyperparameters]):
        if hyperparameters is None:
            hyperparameters = XGBoostHyperparameters()
        self.model = XGBClassifier(**hyperparameters.model_dump())

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        inst = cls(None)
        inst.model = model
        return inst

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


# random forest
class RandomForestHyperparameters(BaseModel):
    n_estimators: int = Field(
        default=100, ge=1, le=1000, description="The number of trees in the forest."
    )
    max_depth: int = Field(
        default=6, ge=1, le=10, description="The maximum depth of the tree."
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        description="The minimum samples required to split a node.",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        description="The minimum samples required at a leaf node.",
    )
    max_features: Union[str, int, float] = Field(
        default="sqrt",
        description="Features considered when looking for the best split.",
    )
    bootstrap: bool = Field(
        default=True, description="Whether bootstrap samples are used."
    )
    n_jobs: Optional[int] = Field(default=None, description="Number of parallel jobs.")
    random_state: Optional[int] = Field(default=None, description="Random seed.")


class RandomForestModel:
    def __init__(self, hyperparameters: Optional[RandomForestHyperparameters]):
        if hyperparameters is None:
            hyperparameters = RandomForestHyperparameters()
        self.model = RandomForestClassifier(**hyperparameters.model_dump())

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        inst = cls(None)
        inst.model = model
        return inst

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


# ensemble model
class EnsembleModel:
    NOT_TRAINABLE_STRATEGIES = ["mean", "median", "mode", "weighted"]
    TRAINABLE_STRATEGIES = ["xgboost", "random_forest"]

    def __init__(
        self,
        models: list[Union[XGBoostModel, RandomForestModel]],
        strategy: str,
        hyperparameters: Optional[dict] = None,
    ):
        self.models = models
        self.strategy = strategy
        self.hyperparameters = hyperparameters or {}
        self.model = None

        if strategy == "weighted":
            if "weights" not in self.hyperparameters:
                raise ValueError("weights are required when strategy is 'weighted'")
            if sum(self.hyperparameters["weights"]) != 1:
                raise ValueError(
                    "weights should add up to 1 when strategy is 'weighted'"
                )

        self._warn_extra_hyperparameters()

    def _warn_extra_hyperparameters(self):
        if not self.hyperparameters:
            return
        expected = self._get_expected_hyperparameters()
        extra = set(self.hyperparameters.keys()) - expected
        if extra:
            warnings.warn(
                f"Strategy '{self.strategy}' ignores hyperparameters: {sorted(extra)}. "
                f"Expected: {sorted(expected) if expected else 'none'}"
            )

    def _get_expected_hyperparameters(self) -> set:
        if self.strategy in ["mean", "median", "mode"]:
            return set()
        if self.strategy == "weighted":
            return {"weights"}
        if self.strategy == "xgboost":
            return set(XGBoostHyperparameters.model_fields.keys())
        if self.strategy == "random_forest":
            return set(RandomForestHyperparameters.model_fields.keys())
        return set()

    def get_scores_from_models(self, X):
        return np.column_stack([model.predict(X) for model in self.models])

    def fit(self, X, y):
        if self.strategy in self.NOT_TRAINABLE_STRATEGIES:
            return
        if self.strategy in self.TRAINABLE_STRATEGIES:
            predictions = self.get_scores_from_models(X)
            if self.strategy == "xgboost":
                self.model = XGBoostModel(
                    XGBoostHyperparameters(**self.hyperparameters)
                )
                self.model.fit(predictions, y)
            elif self.strategy == "random_forest":
                self.model = RandomForestModel(
                    RandomForestHyperparameters(**self.hyperparameters)
                )
                self.model.fit(predictions, y)

    def predict(self, X):
        predictions = self.get_scores_from_models(X)
        if self.strategy == "mean":
            return np.round(np.mean(predictions, axis=1)).astype(int)
        elif self.strategy == "median":
            return np.round(np.median(predictions, axis=1)).astype(int)
        elif self.strategy == "mode":
            return np.apply_along_axis(
                lambda row: np.bincount(row.astype(int)).argmax(), 1, predictions
            )
        elif self.strategy == "weighted":
            return np.round(
                np.average(predictions, axis=1, weights=self.hyperparameters["weights"])
            ).astype(int)
        elif self.strategy == "xgboost":
            return self.model.predict(predictions)
        elif self.strategy == "random_forest":
            return self.model.predict(predictions)
        else:
            raise ValueError(f"Strategy {self.strategy} not supported")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
