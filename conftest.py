import numpy as np
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.random((1000, 10))
    y = rng.integers(0, 2, 1000)
    return train_test_split(X, y, test_size=0.2, random_state=42)
