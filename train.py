from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from algorithms import (
    EnsembleModel,
    RandomForestHyperparameters,
    RandomForestModel,
    XGBoostHyperparameters,
    XGBoostModel,
)


class Config(BaseModel):
    model: Literal["xgboost", "random_forest", "ensemble"]
    strategy: Optional[str] = Field(default=None)
    hyperparameters: dict = Field(default={})
    base_models: Optional[list["Config"]] = Field(default=None)
    train: bool = Field(default=True)
    model_path: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_ensemble_requires_strategy(self):
        if self.model == "ensemble" and self.strategy is None:
            raise ValueError("strategy is required when model is 'ensemble'")
        return self

    @model_validator(mode="after")
    def validate_ensemble_requires_base_models(self):
        if self.model == "ensemble" and self.base_models is None:
            raise ValueError("base_models is required when model is 'ensemble'")
        return self

    @model_validator(mode="after")
    def validate_not_train_requires_model_path(self):
        if not self.train and self.model_path is None:
            raise ValueError("model_path is required when train is False")
        return self


def train_from_config(config, data=None):
    train_config = config if isinstance(config, Config) else Config(**config)
    if train_config.model == "ensemble":
        models = []
        for base in train_config.base_models or []:
            if base.train:
                models.append(train_from_config(base, data))
            elif base.model_path:
                models.append(train_from_config(base))
        if not models:
            raise ValueError("No base models loaded or trained")
        model = EnsembleModel(
            models,
            train_config.strategy,
            train_config.hyperparameters,
        )
        X_train, y_train = data
        model.fit(X_train, y_train)
        return model
    if train_config.train:
        if train_config.model == "xgboost":
            model = XGBoostModel(XGBoostHyperparameters(**train_config.hyperparameters))
        elif train_config.model == "random_forest":
            model = RandomForestModel(
                RandomForestHyperparameters(**train_config.hyperparameters)
            )
        else:
            raise ValueError(f"Model {train_config.model} not supported")
        X_train, y_train = data
        model.fit(X_train, y_train)
        return model
    if train_config.model == "xgboost":
        return XGBoostModel.load(train_config.model_path)
    if train_config.model == "random_forest":
        return RandomForestModel.load(train_config.model_path)
    if train_config.model == "ensemble":
        return EnsembleModel.load(train_config.model_path)
    raise ValueError(f"Model {train_config.model} not supported")
