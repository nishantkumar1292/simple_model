import os
import tempfile

import numpy as np
import pytest

from algorithms import (
    EnsembleModel,
    RandomForestHyperparameters,
    RandomForestModel,
    XGBoostHyperparameters,
    XGBoostModel,
)
from train import train_from_config


def test_xgboost_model(data):
    X_train, X_test, y_train, y_test = data
    model = XGBoostModel(XGBoostHyperparameters())
    model.fit(X_train, y_train)
    assert model.predict(X_test).shape[0] == y_test.shape[0]


def test_random_forest_model(data):
    X_train, X_test, y_train, y_test = data
    model = RandomForestModel(RandomForestHyperparameters())
    model.fit(X_train, y_train)
    assert model.predict(X_test).shape[0] == y_test.shape[0]


def test_train_single_model_from_config(data):
    config = {
        "model": "xgboost",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    }
    X_train, X_test, y_train, y_test = data
    model = train_from_config(config, (X_train, y_train))
    assert model.predict(X_test).shape[0] == y_test.shape[0]


def test_load_xgboost_model_from_config(data):
    # train xgboost model and save as temporary file
    X_train, _, y_train, _ = data
    model = XGBoostModel(XGBoostHyperparameters())
    model.fit(X_train, y_train)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        path = temp_file.name
    try:
        model.save(path)
        config = {"model": "xgboost", "train": False, "model_path": path}
        model = train_from_config(config)
        assert model is not None
    finally:
        os.remove(path)


def test_train_ensemble_model_from_config(data):
    # train random forest model and save as temporary file
    X_train, _, y_train, _ = data
    model = RandomForestModel(RandomForestHyperparameters(max_depth=8))
    model.fit(X_train, y_train)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
        path = temp_file.name
        model.save(path)
        try:
            # train ensemble model
            config = {
                "model": "ensemble",
                "strategy": "xgboost",
                "base_models": [
                    {
                        "model": "xgboost",
                        "hyperparameters": {
                            "n_estimators": 100,
                            "max_depth": 6,
                            "learning_rate": 0.1,
                            "subsample": 1.0,
                            "colsample_bytree": 1.0,
                            "gamma": 0.0,
                            "reg_alpha": 0.0,
                            "reg_lambda": 1.0,
                        },
                    },
                    {
                        "model": "random_forest",
                        "train": False,
                        "model_path": path,
                    },
                ],
                "hyperparameters": {"max_depth": 8},
            }
            X_train, X_test, y_train, y_test = data
            model = train_from_config(config, (X_train, y_train))
            assert model.model is not None
            assert model.predict(X_test).shape[0] == y_test.shape[0]
        finally:
            os.remove(path)


class TestEnsembleModelWeightedStrategy:
    def test_weighted_strategy(self, data):
        X_train, _, y_train, _ = data
        config = {
            "model": "ensemble",
            "strategy": "weighted",
            "hyperparameters": {"weights": [0.6, 0.4]},
            "base_models": [
                {"model": "xgboost"},
                {"model": "random_forest"},
            ],
        }
        model = train_from_config(config, (X_train, y_train))
        assert model.predict(X_train).shape[0] == y_train.shape[0]

    def test_random_forest_strategy_with_none_hyperparameters(self, data):
        X_train, X_test, y_train, y_test = data
        xgb = XGBoostModel(XGBoostHyperparameters())
        xgb.fit(X_train, y_train)
        rf = RandomForestModel(RandomForestHyperparameters())
        rf.fit(X_train, y_train)
        ensemble = EnsembleModel(
            [xgb, rf], strategy="random_forest", hyperparameters=None
        )
        # This should work but will raise TypeError
        ensemble.fit(X_train, y_train)


class TestNestedEnsembleCannotLoad:
    """EnsembleModel has no load() method, so nested ensembles with train=False fail."""

    def test_ensemble_has_no_save_method(self, data):
        X_train, X_test, y_train, y_test = data
        xgb = XGBoostModel(XGBoostHyperparameters())
        xgb.fit(X_train, y_train)
        rf = RandomForestModel(RandomForestHyperparameters())
        rf.fit(X_train, y_train)
        ensemble = EnsembleModel([xgb, rf], strategy="mean")
        # EnsembleModel should have save() like other models
        assert hasattr(ensemble, "save"), "EnsembleModel should have a save method"

    def test_ensemble_has_no_load_method(self):
        # EnsembleModel should have load() like other models
        assert hasattr(EnsembleModel, "load"), (
            "EnsembleModel should have a load class method"
        )

    def test_nested_ensemble_config_with_pretrained_ensemble_fails(self, data):
        """This would fail in train.py because there's no handling for loading ensemble models."""
        X_train, X_test, y_train, y_test = data
        # Even if we could save an ensemble, the train_from_config doesn't handle loading it
        config = {
            "model": "ensemble",
            "strategy": "mean",
            "base_models": [
                {"model": "xgboost"},
                {
                    "model": "ensemble",  # Nested ensemble
                    "train": False,
                    "model_path": "/fake/path.pkl",  # Would fail - no load support
                    "strategy": "mean",
                    "base_models": [],  # Required by validator
                },
            ],
        }
        # This should be able to load a pre-trained ensemble, but it can't
        with pytest.raises((ValueError, AttributeError, FileNotFoundError)):
            train_from_config(config, (X_train, y_train))


class TestModeStrategyPredictions:
    """mode strategy uses np.bincount which only works with non-negative integers."""

    def test_mode_strategy_with_float_predictions_conceptual(self, data):
        """
        This test demonstrates that mode strategy assumes integer predictions.
        While sklearn classifiers return integers, the astype(int) can cause issues
        if models return floats or negative values.
        """
        X_train, X_test, y_train, y_test = data
        xgb = XGBoostModel(XGBoostHyperparameters())
        xgb.fit(X_train, y_train)
        rf = RandomForestModel(RandomForestHyperparameters())
        rf.fit(X_train, y_train)
        ensemble = EnsembleModel([xgb, rf], strategy="mode")
        # This works for classification with 0/1 labels, but the implementation is fragile
        result = ensemble.predict(X_test)
        # Verify output is integer type (the mode impl converts to int)
        assert result.dtype in [np.int64, np.int32, int]


class TestEnsembleTrainingWithNoneData:
    """train_from_config unpacks data without checking if it's None."""

    def test_ensemble_training_with_none_data_fails(self):
        config = {
            "model": "ensemble",
            "strategy": "mean",
            "base_models": [
                {"model": "xgboost"},
                {"model": "random_forest"},
            ],
        }
        # This should raise a clear error, but instead raises:
        # TypeError: cannot unpack non-iterable NoneType object
        with pytest.raises(TypeError, match="cannot unpack non-iterable NoneType"):
            train_from_config(config, data=None)

    def test_single_model_training_with_none_data_fails(self):
        config = {"model": "xgboost"}
        # Same issue for single models
        with pytest.raises(TypeError, match="cannot unpack non-iterable NoneType"):
            train_from_config(config, data=None)


class TestInconsistentPredictionTypes:
    """Different strategies return different types (float vs int)."""

    def test_mean_returns_float_median_returns_float_mode_returns_int(self, data):
        X_train, X_test, y_train, y_test = data
        xgb = XGBoostModel(XGBoostHyperparameters())
        xgb.fit(X_train, y_train)
        rf = RandomForestModel(RandomForestHyperparameters())
        rf.fit(X_train, y_train)

        mean_ensemble = EnsembleModel([xgb, rf], strategy="mean")
        median_ensemble = EnsembleModel([xgb, rf], strategy="median")
        mode_ensemble = EnsembleModel([xgb, rf], strategy="mode")

        mean_pred = mean_ensemble.predict(X_test)
        median_pred = median_ensemble.predict(X_test)
        mode_pred = mode_ensemble.predict(X_test)

        # These should all return consistent types for classification
        # Currently: mean/median return float, mode returns int
        assert mean_pred.dtype == median_pred.dtype == mode_pred.dtype, (
            f"Inconsistent prediction types: mean={mean_pred.dtype}, "
            f"median={median_pred.dtype}, mode={mode_pred.dtype}"
        )
