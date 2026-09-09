#!/usr/bin/env python
# coding: utf-8

"""
Script for training a CNN segmentation model based on GeoTiff data and label files,
using the Clay v1 geospatial foundation model (via TerraTorch) as a pretrained backbone.

Classes: fields, forest, sea, urban (4 classes)
Input bands (10, in this exact order): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

The main Python libraries are PyTorch, PyTorch Lightning, TorchGeo, and TerraTorch.
Main steps of the script:
* Data loading (TorchGeo, RandomGeoSampler / GridGeoSampler over large untiled scenes)
* Augmentation (Kornia, GPU-side)
* Model training (Clay backbone + UNet decoder, via TerraTorch's SemanticSegmentationTask)

Created on Fri Oct 3 2025
@author: ihakulin, kylliek

Ideas and codesnippets from:
* https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
* https://medium.com/@geografif/geospatial-deep-learning-using-torchgeo-and-custom-datasets-2adae17f2df4
"""

import os, sys, time, datetime
from typing import Any, Dict, List

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# PyTorch lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import LightningDataModule

# TorchGeo
from torchgeo.datasets import RasterDataset, BoundingBox, UnionDataset
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# TerraTorch — provides the Clay backbone + segmentation task wrapper
from terratorch.tasks import SemanticSegmentationTask

# Data augmentation
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K


def create_union_dataset(image_dir, mask_dir):
    """
    Build a combined image+mask TorchGeo dataset.

    Input GeoTIFFs are expected to already contain exactly these 10 bands,
    in this exact order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    """
    class Image(RasterDataset):
        filename_glob = "*.tif" # CHANGED
        is_image = True
        all_bands = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")

    class Mask(RasterDataset):
        filename_glob = "*.tif" # CHANGED
        is_image = False

    return UnionDataset(
        Image(paths=image_dir),   # CHANGED: paths=image_dir instead of Image(".")
        Mask(paths=mask_dir)      # CHANGED: paths=mask_dir instead of Mask(".")
    )


class GeoDataModule(LightningDataModule):
    """
    A TorchGeo GeoDataModule for loading imagery and labels, and creating an
    iterable Torch Dataloader over the training data.

    Uses RandomGeoSampler for training (random crops, refreshed each epoch)
    and GridGeoSampler for validation (deterministic, non-overlapping coverage).
    """

    def __init__(self, train_images, train_masks, val_images, val_masks,
                 tile_size, batch_size, num_workers, sampler_length):
        super().__init__()
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_length = sampler_length

    def setup(self, stage=None):
        self.train_dataset = create_union_dataset(self.train_images, self.train_masks)
        self.val_dataset = create_union_dataset(self.val_images, self.val_masks)
        self.train_sampler = RandomGeoSampler(self.train_dataset, size=self.tile_size, length=self.sampler_length)
        self.val_sampler = GridGeoSampler(self.val_dataset, size=self.tile_size, stride=self.tile_size // 2)

    def collate_fn(self, batch):
        # Keep only image/mask — TerraTorch's task forwards every other batch key
        # straight into the model as a kwarg, so geo metadata (bounds/crs/transform)
        # must not be included here.
        images = torch.stack([item["image"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        return {"image": images, "mask": masks.long()}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


class MySegmentationTask(SemanticSegmentationTask):
    """
    TerraTorch's SemanticSegmentationTask, extended to run Kornia data
    augmentation on the GPU inside on_after_batch_transfer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = None

    def augment_training_data(self):
        aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90, resample="nearest"),
            data_keys=["image", "mask"],
            keepdim=True
        )
        return aug

    def on_after_batch_transfer(self, batch: Dict[str, torch.Tensor], dataloader_idx: int) -> Dict[str, torch.Tensor]:
        if self.trainer.training:
            if self.aug is None:
                self.aug = self.augment_training_data()
            self.aug.to(self.device)
            images = batch["image"].float()
            masks = batch["mask"].long()
            images_aug, masks_aug = self.aug(images, masks)
            batch["image"] = images_aug
            batch["mask"] = masks_aug.long()
        return batch


def train_model(lightning_model, datamodule, no_of_epochs, patience, logs_dir, checkpoints_dir):
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoints_dir, filename="best_model",
        monitor="val/mIoU",   # CHANGED: was "val_loss"
        save_top_k=1, 
        mode="max" # CHANGED: was "min"
    )
    earlystop_cb = EarlyStopping(
        monitor="val/mIoU",   # CHANGED: was "val_loss"
        patience=patience, 
        mode="max" # CHANGED: was "min"
    )  
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="segmentation")

    trainer = pl.Trainer(
        max_epochs=no_of_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16",
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=tb_logger,
        log_every_n_steps=10,
    )
    trainer.fit(lightning_model, datamodule=datamodule)


def main():
    base_folder = os.path.join(os.sep, 'scratch', 'project_2019932', 'students', os.environ.get('USER'), 'GeoML_old2')
    exercise_folder = os.path.join(base_folder, '07_cnn_segmentation', '07B_using_foundation_model')
    cnn_data_folder = os.path.join(base_folder, 'data', 'raster', 'cnn')
    logs_dir = os.path.join(exercise_folder, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoints_dir = os.path.join(exercise_folder, 'checkpoints')

    data_train_folder = os.path.join(cnn_data_folder, 'train', 'data')
    data_validation_folder = os.path.join(cnn_data_folder, 'validation', 'data')
    labels_train_folder = os.path.join(cnn_data_folder, 'train', 'labels')
    labels_validation_folder = os.path.join(cnn_data_folder, 'validation', 'labels')

    # --- Model / training settings ---
    num_classes = 4                 # fields, forest, sea, urban
    loss = 'focal'                   # currently supports ‘ce’, ‘bce’, ‘jaccard’, ‘focal’, and ‘dice’ loss.
    learning_rate = 1e-3
    patience = 10                  # CHANGED to save time
    freeze_backbone = True         # set True to only fine-tune the decoder (useful with limited labeled data)

    batch_size = 8
    num_cpus = len(os.sched_getaffinity(0))
    tile_size = 224                 # must match backbone_img_size below
    sampler_length = 1600
    num_epochs = 200

    datamodule = GeoDataModule(
        train_images=data_train_folder,
        train_masks=labels_train_folder,
        val_images=data_validation_folder,
        val_masks=labels_validation_folder,
        tile_size=tile_size,
        batch_size=batch_size,
        num_workers=num_cpus,
        sampler_length=sampler_length
    )

    model = MySegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "clay_v1_base",
            "backbone_pretrained": True,
            "backbone_img_size": tile_size,   # required — Clay defaults to 256 internally otherwise
            "backbone_bands": [
                "BLUE", "GREEN", "RED",
                "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
                "NIR_BROAD", "NIR_NARROW",
                "SWIR_1", "SWIR_2",
            ],  # must match the band order in create_union_dataset exactly
            "necks": [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},  # Clay has 12 transformer layers (0-11)
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "num_classes": num_classes,
        },
        loss=loss,
        optimizer="AdamW",
        lr=learning_rate,
        ignore_index=-100, #CHANGED from None
        freeze_backbone=freeze_backbone,
        plot_on_val=False,   # NEW — disables datamodule.plot() calls, which your custom GeoDataModule doesn't implement        
    )

    train_model(model, datamodule, num_epochs, patience, logs_dir, checkpoints_dir)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start) / 60), 0)) + " minutes")