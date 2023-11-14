from typing import Dict

import datajoint as dj
from ..main import GenerateDataset
from ...generators import resolve_generator

from nnfabrik.utility.nnf_helper import cleanup_numpy_scalar


class RenderedImagesBase(dj.Computed):
    """
    Base class for defining a RenderedImage table used to store individual rendered images, along with their
    image latents and other meta-data about the image.

    To use this class, define a new class inheriting from this base class, and decorate with your own
    schema.

    This table should depend on an ImageConfig table, that stores the configs of individual images.
        Per default, it depends on the GenerateDataset.ImageConfig table, as found in ...schema.main
    """

    # table level comment
    table_comment = "rendered images"

    image_config_table = GenerateDataset.ImageConfig

    @property
    def definition(self):
        definition = """
        # {table_comment}
        -> self.image_config_table()
        ---
        image:                             longblob     # actual image, np.ndarray (c, h, w) with c = channels
        latents:                           longblob     # image latents, np.ndarray (n, h, w) with n = number of latents
        meta_data:                         longblob     # other form of meta data
        rendering_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(
            table_comment=self.table_comment
        )
        return definition

    def get_generator_fn_config(self, key: Dict = None):
        if key is None:
            key = {}
        generator_fn, image_config = (self.image_config_table() & key).fetch1(
            "generator_fn", "image_config"
        )
        generator_fn = resolve_generator(generator_fn)
        image_config = cleanup_numpy_scalar(image_config)
        return generator_fn, image_config

    def make(self, key):
        """
        Given key specifying the image generation function and config, this function will render the image,
        and store the image along with latents and meta data
        """

        generator_fn, image_config = self.get_generator_fn_config(key)
        image, latents, meta_data = generator_fn(**image_config)

        # typechecks
        if image.dtype.name != "uint8":
            raise ValueError(
                "The rendered images are not in 8bit format. "
                "Images are expected to be of type (np.ndarray) of dtype np.uint8"
            )

        if latents.dtype.name != "uint8":
            raise ValueError(
                "The latents are not in 8bit format. Latents are expected to be of type (np.ndarray) and dtype np.uint8"
            )

        key["image"] = image
        key["latents"] = latents
        key["meta_data"] = meta_data
        self.insert1(key)
