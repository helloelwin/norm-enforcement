import json
import random


def load_dataset_with_seed(
    dataset_path: str,
    max_samples: int,
    seed: int,
) -> list[dict]:
    """
    Load dataset from a JSON file.
    Randomly sample entries based on the seed for reproducibility.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    if len(data) > max_samples:
        return rng.sample(data, max_samples)
    else:
        # Shuffle to ensure randomness even when using all data
        shuffled = data.copy()
        rng.shuffle(shuffled)
        return shuffled
