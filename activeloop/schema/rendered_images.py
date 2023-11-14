import os
import tempfile

from .templates.rendered_images import RenderedImagesBase
from .main import GenerateDataset
from . import schema


@schema
class RenderedImages(RenderedImagesBase):
    image_config_table = GenerateDataset.ImageConfig


# example custom RenderedImagetable
class MyCustomRenderedImageTable(RenderedImagesBase):
    """
    to define a custom table, simply inherit from the Base class.
    Now you can, for example:
        1. Extend the definition to include more table attributes
        and
        2. choose a different image table
        and
        3. overwrite the make function to fill the new attributes
    """
    image_config_table = GenerateDataset.ImageConfig
    external_storage = "some_s3_storage"

    # new definition to get more table attributes
    @property
    def definition(self):
        definition = """
            # {table_comment}
            -> self.image_config_table()
            ---
            image:                             longblob     # actual image, np.ndarray (c, h, w) with c = channels
            latents:                           longblob     # image latents, np.ndarray (n, h, w) with n = number of latents
            meta_data:                         longblob     # other form of meta data
            new_attribute:                     longblob     # some other image details
            blender_file:                      attach@external_storage
            rendering_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(
            table_comment=self.table_comment
        )
        return definition

    # new make function
    def make(self, key):
        """
        Given key specifying the image generation function and config, this function will render the image,
        and store the image along with latents and meta data
        """

        generator_fn, image_config = self.get_generator_fn_config(key)

        # getting more arguments returned from the generator
        image, latents, meta_data, more_data, blender_filename = generator_fn(**image_config)

        # other typechecks, or no typechecks
        if latents.dtype.name != "float32":
            raise ValueError(
                "The latents are not in 32bit format. Latents are expected to be of type (np.ndarray) and dtype np.float32"
            )

        key["image"] = image
        key["latents"] = latents
        key["meta_data"] = meta_data

        # add new attributes
        key["new_attribute"] = more_data

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, blender_filename)
            key["model_state"] = filepath # attaching the file to the external storage
            self.insert1(key)

