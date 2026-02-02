import yaml
from sklearn.datasets import make_classification

from train import train_from_config

# Load the doomed config
with open("doomed_config.yaml") as f:
    config = yaml.safe_load(f)

# Generate some dummy data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Attempt training (this will fail spectacularly)
print("Attempting to train from doomed_config.yaml...")
model = train_from_config(config, data=(X, y))
print("Successfully trained model!")
