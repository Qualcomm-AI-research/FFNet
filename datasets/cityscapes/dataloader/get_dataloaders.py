# import datasets.cityscapes.dataloader.joint_transforms as joint_transforms
import datasets.cityscapes.dataloader.transforms as extended_transforms
from torch.utils.data import DataLoader

import importlib
import torchvision.transforms as standard_transforms
from config import CITYSCAPES_MEAN, CITYSCAPES_STD
from datasets.cityscapes.dataloader.base_loader import Cityscapes


def return_dataloader(num_workers, batch_size):
    """
    Return Dataloader
    """

    val_joint_transform_list = None

    mean_std = (CITYSCAPES_MEAN, CITYSCAPES_STD)
    target_transform = extended_transforms.MaskToTensor()

    val_input_transform = standard_transforms.Compose(
        [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
    )

    val_frame = Cityscapes(
        mode="val",
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=None,
    )

    # if cfg.apex:
    #    from library.datasets.sampler import DistributedSampler
    #    val_sampler = DistributedSampler(val_frame, pad=False, permutation=False,
    #                                     consecutive_sample=False)
    # else:
    val_sampler = None

    val_loader = DataLoader(
        val_frame,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
    )

    return val_loader
