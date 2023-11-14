import numpy as np
from typing import Callable, Optional, Tuple, Union, Dict, List

import random


def demo_dataset(
    seed: int = 0,
    n_images: int = 100,
    image_size: Tuple[int, int] = (100, 100),
    rotations: Tuple[int, int] = (-45, 45),
    objects: List[str] = None,
) -> List[Dict]:
    """
    A dataset returns a list of dictionaries, each dictionary is the config for a single image.

    The dataset thus contains all the images that will be subsequently generated.
    The list of config dictionaries will be passed to the ImageConfig table, and from there it will be rendered.

    So, note that this function will not generate images yet, but it will only create the configs for the images.

    Args:
        seed: Sets the random seed. Make sure to always set a seed for reproducibility.
        n_images: number of images
        image_size: (height, width)
        rotations: (min, max) rotation in degrees
        objects: list of objects to use

    Returns:
        A list of dictionaries, each dictionary is the config for a single image.
    """

    random.seed(seed)
    np.random.seed(seed)

    dataset = []
    for i in range(n_images):
        dataset.append(
            {
                "image_size": image_size,
                "rotation": np.random.randint(rotations[0], rotations[1]),
                "object": random.choice(objects),
            }
        )

    return dataset
