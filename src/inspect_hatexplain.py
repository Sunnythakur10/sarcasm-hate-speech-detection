import json
import os
import numpy as np


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "HateXplain",
    "Data"
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "dataset.json"
)

CLASSES_PATH = os.path.join(
    DATA_DIR,
    "classes.npy"
)

DIVISIONS_PATH = os.path.join(
    DATA_DIR,
    "post_id_divisions.json"
)


print("Loading classes...")

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Classes:")
print(classes)


print("\nLoading split information...")

with open(
    DIVISIONS_PATH,
    "r",
    encoding="utf-8"
) as f:
    divisions = json.load(f)

print(
    "Train IDs:",
    len(divisions["train"])
)

print(
    "Validation IDs:",
    len(divisions["val"])
)

print(
    "Test IDs:",
    len(divisions["test"])
)


print("\nLoading dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

print(
    "Total posts:",
    len(data)
)


# Inspect one example
first_id = next(iter(data))

print("\nExample ID:")
print(first_id)

print("\nExample structure:")
print(
    json.dumps(
        data[first_id],
        indent=2
    )[:3000]
)