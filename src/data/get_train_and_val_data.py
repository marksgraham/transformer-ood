from pathlib import Path

import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset


def get_data_dicts(ids_path: str, shuffle: bool = False):
    if Path(ids_path).is_dir():
        train_data_list = list(Path(ids_path).glob("*.nii*"))
        data_dicts = [{"image": d} for d in train_data_list]
    else:
        df = pd.read_csv(ids_path, sep=",")
        df = list(df)
        data_dicts = []
        for row in df:
            data_dicts.append({"image": row})
    if shuffle:
        data_dicts = shuffle(data_dicts)

    print(f"Found {len(data_dicts)} subjects.")
    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts


def get_training_data_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=True,
    spatial_dimension=3,
    is_grayscale=True,
    image_size=None,
    image_roi=None,
    pixel_space=False,
):
    resize_transform = (
        transforms.ResizeD(keys=["image"], spatial_size=(image_size,) * spatial_dimension)
        if image_size
        else lambda x: x
    )

    central_crop_transform = (
        transforms.CenterSpatialCropD(keys=["image"], roi_size=image_roi)
        if image_roi
        else lambda x: x
    )

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]) if is_grayscale else lambda x: x,
            transforms.Lambdad(keys="image", func=lambda x: x[0, None, ...])
            if is_grayscale
            else lambda x: x,  # needed for BRATs data with 4 modalities in 1
            central_crop_transform,
            resize_transform,
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SqueezeDimD(keys=["image"], dim=0) if pixel_space else lambda x: x,
        ]
    )

    val_transforms = train_transforms

    val_dicts = get_data_dicts(validation_ids, shuffle=False)
    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )
    print(val_ds[0]["image"].shape)
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(training_ids, shuffle=False)

    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    return train_loader, val_loader
