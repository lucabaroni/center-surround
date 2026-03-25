import ast
import csv
import hashlib
import os
import pickle
from collections.abc import Iterable

import numpy as np
from nnvision.datasets.utility import (
    ImageCache,
    get_cached_loader_extended,
    get_crop_from_stimulus_location,
    get_validation_split,
)


def _read_responses_csv(csv_path):
    image_ids = []
    responses = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_ids.append(int(row["image_id"]))
            responses.append(ast.literal_eval(row["responses"]))

    if not responses:
        raise ValueError(f"No responses found in {csv_path}")

    return np.asarray(image_ids, dtype=np.int64), np.asarray(responses, dtype=np.float32)


def _get_test_session_dir(train_session_dir):
    train_session_dir = os.path.abspath(train_session_dir)
    if os.path.basename(os.path.dirname(train_session_dir)) != "train":
        raise ValueError(
            "Expected each entry in `neuronal_data_files` to be a train session directory like "
            "`.../train/<session_id>`"
        )

    test_session_dir = os.path.join(
        os.path.dirname(os.path.dirname(train_session_dir)),
        "test",
        os.path.basename(train_session_dir),
    )
    if not os.path.exists(test_session_dir):
        raise FileNotFoundError(
            f"Could not find matching test session directory for {train_session_dir}: "
            f"{test_session_dir}"
        )
    return test_session_dir


def _center_cache_images(cache):
    images = cache.loaded_images
    shift = (images.max() + images.min()) / 2
    for key in list(cache.cache.keys()):
        cache.cache[key] = cache.cache[key] - shift


def _make_hash(value):
    return hashlib.md5(repr(value).encode("utf-8")).hexdigest()


def monkey_static_loader_combined_new_format(
    dataset,
    neuronal_data_files,
    image_cache_path,
    batch_size=64,
    seed=None,
    train_frac=0.8,
    subsample=1,
    crop=((96, 96), (96, 96)),
    scale=1.0,
    time_bins_sum=tuple(range(12)),
    avg=False,
    image_file=None,
    return_data_info=False,
    store_data_info=True,
    image_frac=1.0,
    image_selection_seed=None,
    randomize_image_selection=True,
    img_mean=None,
    img_std=None,
    stimulus_location=None,
    monitor_scaling_factor=4.57,
    include_prev_image=False,
    include_trial_id=False,
    include_bools=True,
    include_n_neurons=False,
    normalize_resps=False,
    center_inputs=False,
):
    if include_prev_image:
        raise NotImplementedError("include_prev_image is not implemented for the new folder format loader")
    if include_trial_id:
        raise NotImplementedError("include_trial_id is not implemented for the new folder format loader")
    if image_frac != 1.0:
        raise NotImplementedError("image_frac is not implemented for the new folder format loader")
    if image_selection_seed is not None or randomize_image_selection is not True:
        raise NotImplementedError(
            "image selection options are not implemented for the new folder format loader"
        )
    if image_file is not None:
        raise NotImplementedError("image_file is not used for the new folder format loader")

    dataset_config = locals().copy()
    dataloaders = {"train": {}, "validation": {}, "test": {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if stimulus_location is not None:
        crop = get_crop_from_stimulus_location(
            stimulus_location, crop, monitor_scaling_factor=monitor_scaling_factor
        )

    image_cache_path = image_cache_path.split("individual")[0]
    stats_filename = _make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, "statistics", stats_filename)

    if os.path.exists(stats_path):
        with open(stats_path, "rb") as handle:
            data_info = pickle.load(handle)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]
        cache = ImageCache(
            path=image_cache_path,
            subsample=subsample,
            crop=crop,
            scale=scale,
            img_mean=img_mean,
            img_std=img_std,
            transform=True,
            normalize=True,
        )
    else:
        if img_mean is not None:
            cache = ImageCache(
                path=image_cache_path,
                subsample=subsample,
                crop=crop,
                scale=scale,
                img_mean=img_mean,
                img_std=img_std,
                transform=True,
                normalize=True,
            )
        else:
            cache = ImageCache(
                path=image_cache_path,
                subsample=subsample,
                crop=crop,
                scale=scale,
                transform=True,
                normalize=False,
            )
            cache.zscore_images(update_stats=True)
            img_mean = cache.img_mean
            img_std = cache.img_std

    if center_inputs:
        _center_cache_images(cache)

    n_images = len(cache)
    if dataset == "PlosCB19_V1":
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(
        n_images=n_images * train_test_split, train_frac=train_frac, seed=seed
    )

    session_data = []
    n_neurons = np.zeros(len(neuronal_data_files), dtype=np.uint32)
    max_repeats = 0
    all_testing_ids = np.array([], dtype=np.uint32)

    for i, train_session_dir in enumerate(neuronal_data_files):
        test_session_dir = _get_test_session_dir(train_session_dir)
        train_image_ids, responses_train = _read_responses_csv(
            os.path.join(train_session_dir, "responses.csv")
        )
        test_image_ids, responses_test = _read_responses_csv(
            os.path.join(test_session_dir, "responses.csv")
        )

        session_data.append((train_image_ids, responses_train, test_image_ids, responses_test))
        n_neurons[i] = responses_train.shape[1]

        testing_image_ids = test_image_ids - image_id_offset
        max_repeats = max(
            np.max(np.unique(testing_image_ids, return_counts=True)[1]),
            max_repeats,
        )
        all_testing_ids = np.unique(np.concatenate((all_testing_ids, testing_image_ids)))

    all_responses_train = np.zeros((len(all_train_ids), np.sum(n_neurons)), dtype=np.float32)
    all_responses_val = np.zeros((len(all_validation_ids), np.sum(n_neurons)), dtype=np.float32)
    all_train_bools = np.full((len(all_train_ids), np.sum(n_neurons)), False)
    all_val_bools = np.full((len(all_validation_ids), np.sum(n_neurons)), False)

    all_testing_ids_unique = all_testing_ids
    all_testing_ids = np.repeat(all_testing_ids, max_repeats)
    all_responses_test = np.zeros((len(all_testing_ids), np.sum(n_neurons)), dtype=np.float32)
    all_test_bools = np.full((len(all_testing_ids), np.sum(n_neurons)), False)

    for i, (train_image_ids, responses_train, test_image_ids, responses_test) in enumerate(session_data):
        training_image_ids = train_image_ids - image_id_offset
        testing_image_ids = test_image_ids - image_id_offset

        if responses_test.ndim == 3:
            if time_bins_sum is not None:
                responses_train = (np.mean if avg else np.sum)(
                    responses_train[:, :, time_bins_sum], axis=-1
                )
                responses_test = (np.mean if avg else np.sum)(
                    responses_test[:, :, time_bins_sum], axis=-1
                )
        elif responses_test.ndim != 2:
            raise ValueError(
                f"Unexpected response tensor rank {responses_test.ndim} in session "
                f"{neuronal_data_files[i]}"
            )

        if normalize_resps:
            responses_mean = responses_train.mean(axis=0, keepdims=True)
            responses_std = responses_train.std(axis=0, keepdims=True)
            responses_std[responses_std == 0] = 1
            responses_train = (responses_train - responses_mean) / responses_std
            responses_test = (responses_test - responses_mean) / responses_std

        n_start = np.sum(n_neurons[0:i])
        n_end = np.sum(n_neurons[0 : i + 1])

        for k, train_id in enumerate(all_train_ids):
            if train_id in training_image_ids:
                j = np.where(train_id == training_image_ids)[0]
                if len(j) > 1:
                    j = j[0]
                all_responses_train[k][n_start:n_end] = responses_train[j]
                all_train_bools[k][n_start:n_end] = True

        for k, val_id in enumerate(all_validation_ids):
            if val_id in training_image_ids:
                j = np.where(val_id == training_image_ids)[0]
                if len(j) > 1:
                    j = j[0]
                all_responses_val[k][n_start:n_end] = responses_train[j]
                all_val_bools[k][n_start:n_end] = True

        for k, test_id in enumerate(all_testing_ids_unique):
            idxs = np.where(test_id == testing_image_ids)[0]
            for j, idx in enumerate(idxs):
                all_responses_test[k * max_repeats + j][n_start:n_end] = responses_test[idx]
                all_test_bools[k * max_repeats + j][n_start:n_end] = True

    all_responses_train = all_responses_train[~(~all_train_bools).all(axis=1)]
    all_train_ids = all_train_ids[~(~all_train_bools).all(axis=1)]
    all_train_bools = all_train_bools[~(~all_train_bools).all(axis=1)]

    all_responses_val = all_responses_val[~(~all_val_bools).all(axis=1)]
    all_validation_ids = all_validation_ids[~(~all_val_bools).all(axis=1)]
    all_val_bools = all_val_bools[~(~all_val_bools).all(axis=1)]

    all_responses_test = all_responses_test[~(~all_test_bools).all(axis=1)]
    all_testing_ids = all_testing_ids[~(~all_test_bools).all(axis=1)]
    all_test_bools = all_test_bools[~(~all_test_bools).all(axis=1)]

    args_train = [all_train_ids, all_responses_train]
    args_val = [all_validation_ids, all_responses_val]
    args_test = [all_testing_ids, all_responses_test]

    if include_bools:
        args_train.insert(1, all_train_bools)
        args_val.insert(1, all_val_bools)
        args_test.insert(1, all_test_bools)
        if include_n_neurons:
            n_neurons = np.insert(np.cumsum(n_neurons), 0, 0).astype(np.int64)
            args_train.insert(2, n_neurons)
            args_val.insert(2, n_neurons)
            args_test.insert(2, n_neurons)

    train_loader = get_cached_loader_extended(
        *args_train,
        batch_size=batch_size,
        image_cache=cache,
        include_bools=include_bools,
        include_n_neurons=include_n_neurons,
    )
    val_loader = get_cached_loader_extended(
        *args_val,
        batch_size=batch_size,
        image_cache=cache,
        include_bools=include_bools,
        include_n_neurons=include_n_neurons,
    )
    test_loader = get_cached_loader_extended(
        *args_test,
        batch_size=None,
        shuffle=None,
        image_cache=cache,
        repeat_condition=all_testing_ids,
        include_bools=include_bools,
        include_n_neurons=include_n_neurons,
    )

    data_key = "all_sessions"
    dataloaders["train"][data_key] = train_loader
    dataloaders["validation"][data_key] = val_loader
    dataloaders["test"][data_key] = test_loader

    if os.path.exists(stats_path):
        final_data_info = data_info
    else:
        sample_batch = next(iter(train_loader))
        final_data_info = {
            data_key: {
                "input_dimensions": sample_batch.inputs.shape,
                "input_channels": int(sample_batch.inputs.shape[1]),
                "output_dimension": int(sample_batch.targets.shape[-1]),
                "img_mean": img_mean,
                "img_std": img_std,
            }
        }
        if store_data_info:
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            with open(stats_path, "wb") as handle:
                pickle.dump(final_data_info, handle)

    return final_data_info if return_data_info else dataloaders
