from typing import Dict, List, Tuple, Any

import numpy as np
from numpy import ndarray
from scipy import ndimage


def render_image(
    image_size: Tuple[int, int],
    rotation: int,
    object: str,
) -> Tuple[ndarray, ndarray, Dict]:
    """
    This function takes a config of a single image and creates the image.

    Args:
        image_size: (height, width)
        rotation: rotation in degrees
        object: object to include in the image
        image_format: image format of the generated images
            supported formats: "jpg", "png", "npy"

    Returns:
        image: A numpy array representing the image, in 8bit space: [0, 255] as np.unit8
        latents:
    """

    # generate image
    image = np.random.randn(*image_size).astype(np.uint8)

    # apply random transformations
    image = ndimage.rotate(image, rotation, reshape=False)
    if object == "circle":
        image *= 1
    elif object == "square":
        image *= 2
    elif object == "triangle":
        image *= 3
    else:
        raise ValueError(f"Unknown object {object}")

    # generate image latents
    image_latens = np.ones(
        (10, *image.shape)
    ).astype(np.uint8)  #  10 latent dimensions: segmentation map, depth_map, ...

    # come up with the dictionary extra info, that stores info about the images and latens
    extra_info = {
        "number of objects": 1,
        "camera angle": 45,
    }

    return image, image_latens, extra_info
