import warnings
from typing import Union, Optional, MutableMapping, Tuple, Dict

import datajoint as dj

from nnfabrik.builder import resolve_data, get_data
from nnfabrik.utility.nnf_helper import cleanup_numpy_scalar
from nnfabrik.utility.dj_helpers import make_hash

from .. import schema


@schema
class CloudDataset(dj.Manual):
    definition = """
    dataset_fn:                     varchar(64)    # name of the dataset loader function
    dataset_hash:                   varchar(64)    # hash of the configuration object
    dataset_path:                   varchar(265)   # activeloop dataset path
    ---
    dataset_config:                 longblob       # dataset configuration object
    dataset_comment='' :            varchar(256)    # short description
    dataset_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        dataset_fn, dataset_config = self.fetch1("dataset_fn", "dataset_config")
        dataset_config = cleanup_numpy_scalar(dataset_config)
        return dataset_fn, dataset_config

    @staticmethod
    def resolve_fn(fn_name):
        return resolve_data(fn_name)

    def add_entry(
        self,
        dataset_fn,
        dataset_config,
        dataset_path,
        dataset_comment="",
        dataset_fabrikant=None,
        skip_duplicates=False,
        return_pk_only=True,
    ):
        """
        Add a new entry to the dataset.

        Args:
            dataset_fn (string, Callable): name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
            dataset_config (dict): Python dictionary containing keyword arguments for the dataset_fn
            dataset_comment (str, optional):  Comment for the entry. Defaults to "" (emptry string)
            dataset_fabrikant (string): The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based
                on the database user name for the existing connection.
            skip_duplicates (bool, optional): If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found. Defaults to False.
            return_pk_only (bool, optional): If True, only the primary key attribute for the new entry or corresponding existing entry is returned. Otherwise, the entire
                entry is returned. Defaults to True.

        Returns:
            dict: the entry in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """
        if not isinstance(dataset_fn, str):
            # infer the full path to the callable
            dataset_fn = dataset_fn.__module__ + "." + dataset_fn.__name__

        try:
            resolve_data(dataset_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return


        dataset_hash = make_hash(dataset_config)
        key = dict(
            dataset_fn=dataset_fn,
            dataset_hash=dataset_hash,
            dataset_config=dataset_config,
            dataset_fabrikant=dataset_fabrikant,
            dataset_comment=dataset_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        if return_pk_only:
            key = {k: key[k] for k in self.heading.primary_key}

        return key

    def get_dataloader(self, seed=None, key=None):
        """
        Returns a dataloader for a given dataset loader function and its corresponding configurations
        dataloader: is expected to be a dict in the form of
                            {
                            'train': torch.utils.data.DataLoader,
                            'val': torch.utils.data.DataLoader,
                            'test: torch.utils.data.DataLoader,
                             }
                             or a similar iterable object
                each loader should have as first argument the input such that
                    next(iter(train_loader)): [input, responses, ...]
                the input should have the following form:
                    [batch_size, channels, px_x, px_y, ...]
        """
        if key is None:
            key = {}

        dataset_fn, dataset_config = (self & key).fn_config

        if seed is not None:
            dataset_config["seed"] = seed  # override the seed if passed in

        return get_data(dataset_fn, dataset_config)