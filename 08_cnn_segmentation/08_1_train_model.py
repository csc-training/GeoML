#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
Script for training a CNN segmentation model based on GeoTiff data and label files.
The main Python libraries are Pytorch, PyTorch Lightning and Torchgeo.

Main steps of the script:
* Data loading
* Augmentation
* Model training

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
from torchgeo.trainers import SemanticSegmentationTask

# Data augmentation
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K 


# The data contains both imagery and ground truth masks. We want to load both of these rasters and combine  them into a one dataset that can be fed to the neural network. 
# We will first create a TorchGeo RasterDataset of both rasters and then combine them with UnionDataset from TorchGeo. 
# The is_image attribute is used to control how the data stored in the dataset is handled. 
def create_union_dataset(images, labels):
    class Image(RasterDataset):
        filename_glob = images
        is_image = True
        
    class Mask(RasterDataset):
        filename_glob = labels
        is_image = False
        
    return UnionDataset(Image("."), Mask("."))


class GeoDataModule(LightningDataModule):
    """
    A TorchGeo GeoDataModule for loading imagery and labels, and creating an iterabel Torch Dataloader over the training data. The module first loads the training and validation datasets and creates a TorchGeo UnionDataset that combines both the imagery and label rasters. Training data is sampled randomly to in crease the amount of the samples. A collate function is used to batch the data and stack the right objects inside the dataset. Finally, a Dataloader is created. 
    Attributes:
    ----------
    train_image: list[str] - List of .tif files containing training imagery
    train_mask: list[str] - List of .tif files containing training imagery
    val_image: list[str] - List of .tif files containg the validation imagery
    val_mask: list[str] - List of .tif files containg the validation imagery
    tile_size: a float - tile size used for the sampling
    batch_size: an integer - number of samples per batch
    num_workers: an integer - 

    Methods: 
    setup: Set up datasets and samplers.
    collate_fn: stack objects to form batches.
    train_dataloader: Implement Pytorch Dataloader for training.
    val_dataloader: Implement Pytorch Dataloader for validation.

    Returns:
    --------
    A DataLoader
    """
    def __init__(self, train_images, train_masks, val_images, val_masks, tile_size, batch_size, num_workers, sampler_length):
        super().__init__()
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images   = val_images
        self.val_masks    = val_masks
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_length = sampler_length

    def setup(self, stage=None):
        self.train_dataset = create_union_dataset(self.train_images, self.train_masks)
        self.val_dataset = create_union_dataset(self.val_images, self.val_masks)
    
        self.train_sampler = RandomGeoSampler(self.train_dataset, size=self.tile_size, length=self.sampler_length)
        self.val_sampler = GridGeoSampler(self.val_dataset, size=self.tile_size, stride=self.tile_size//2)
    
    def collate_fn(self, batch):
        images = torch.stack([item["image"] for item in batch])
        masks  = torch.stack([item["mask"] for item in batch])                                                        
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

# LightningModule wrapper for SemanticSegmentationTask
class MySegmentationTask(SemanticSegmentationTask):
    """Kornia Augmentation is run using a LightningModule wrapper for TorchGeo's SemanticSegmentationTask to run the augmentation on GPU. 
    """    
    def __init__(self, *args, aug_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_training_data = None
        
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
            # Called after Lightning moves the batch to the device
            if self.trainer.training:
                if self.augment_training_data is None:
                    self.augment_training_data = self.augment_training_data()

                # Ensure augmentation module is on correct device and dtype
                self.augment_training_data.to(device)

                # Kornia expects floats for images; masks should remain integers.
                images = batch["image"].float()
                masks = batch["mask"].long()
                 
                images_aug, masks_aug = self.augment_training_data(images, masks)
                      
                # Ensure mask dtype is integer for loss functions
                batch["image"] = images_aug
                batch["mask"] = masks_aug.long()
            return batch        

#  # Define Pytorch lightning Trainer and train the model
def train_model(lightning_model, datamodule, no_of_epochs, patience, logs_dir, checkpoints_dir):
    # Add checkpoints to the training, only save the best model based on the minimum validation loss
    checkpoint_cb = ModelCheckpoint(dirpath=checkpoints_dir, filename="best_model", monitor="val_loss", save_top_k=1, mode="min")

    # Add earlystopping to prevent model from overfitting by stopping the training 
    # if validation loss doesn't decrease in patience number of epochs
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    # Enable writing of log files for Tensorboard
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="segmentation")

    # Define Lightning trainer using callbacks and logger
    # In case a checkpoint exists, training can be continued with resume_from_checkpoint="checkpoints/last.ckpt"
    trainer = pl.Trainer(
        max_epochs=no_of_epochs, 
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, earlystop_cb],
        log_every_n_steps=10,)

    # Train model
    trainer.fit(lightning_model, datamodule=datamodule)


def main():
    # Set path to data and labels files
    # With small adjustments instead of files, these could be also folders.
    # See: https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html
    base_folder = os.path.join(os.sep, 'scratch', 'project_462001167', 'students', os.environ.get('USER'), 'GeoML')
    exercise_folder = os.path.join(base_folder, '08_cnn_segmentation') 
    data_folder = os.path.join(base_folder,'data', 'raster')
    logs_dir= os.path.join(exercise_folder, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoints_dir= os.path.join(exercise_folder, 'checkpoints')

    data_deep = os.path.join(data_folder, 'data_deep.tif')
    data_validation = os.path.join(data_folder, 'data_validation.tif')
    data_deep

    labels_deep = os.path.join(data_folder, 'labels_deep.tif')
    labels_validation = os.path.join(data_folder, 'labels_validation.tif')

    # Training settings:
    # SemanticSegmentationTask
    # See: https://torchgeo.readthedocs.io/en/stable/api/trainers.html#torchgeo.trainers.SemanticSegmentationTask
    segmentation_model = "unet"
    backbone = "resnet34" # 
    in_channels = 8 # Number of bands in data image 
    num_classes = 4 # Number of classes in the labels data
    loss = 'ce' 
    learning_rate = 1e-3 #
    patience = 20 # How many epochs model training is continued, if loss does not improve any more.

    # Datamodule
    batch_size = 8 #16 or 32 might be better for bigger datasets
    num_cpus = len(os.sched_getaffinity(0))
    tile_size = 512 # Could be also different, for example: 256
    sampler_length = 1600
    num_epochs = 40 # We use a low number on the course, should be higher in actual projects

    # Create GeoDataModule with our data
    datamodule = GeoDataModule(
        train_images=data_deep,
        train_masks=labels_deep,
        val_images=data_validation,
        val_masks=labels_validation,
        tile_size=tile_size,
        batch_size=batch_size,
        num_workers=num_cpus,
        sampler_length=sampler_length
    )

    # Create SegmentationTask
    model = MySegmentationTask(
        model = segmentation_model,
        backbone = backbone,
        weights = None, 
        in_channels = in_channels,
        num_classes = num_classes,
        loss = loss,
        ignore_index = None,
        lr = learning_rate,
        patience = patience, 
    )

    # Train the model. This is the part of code, that can a long time.
    train_model(model, datamodule, num_epochs, patience, logs_dir, checkpoints_dir)



if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 





