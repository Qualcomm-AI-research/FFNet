# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
##############################################################################
# Code adapted from https://github.com/Qualcomm-AI-research/InverseForm
##############################################################################


from __future__ import absolute_import
from __future__ import division
import numpy as np
import torch
import os
import sys
from datasets.cityscapes.utils.misc import AverageMeter, eval_metrics
from datasets.cityscapes.utils.trnval_utils import eval_minibatch
from datasets.cityscapes.utils.progress_bar import printProgressBar
from datasets.cityscapes.dataloader.get_dataloaders import return_dataloader
import warnings

# from config import cityscapes_base_path


from argparse import ArgumentParser
from models.model_registry import model_entrypoint

if not sys.warnoptions:
    warnings.simplefilter("ignore")

torch.backends.cudnn.benchmark = True


def opts_parser():
    usage = "Set the dataset path in config.py"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        "--model_name",
        type=str,
        default="segmentation_ffnet122NS_CBB",
        metavar="MODEL_NAME",
        help="Select the model configuration",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to model weights",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use.")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size of the input"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        # type=bool,
        # default=False,
        help="Use half precision for faster inference",
    )
    parser.add_argument(
        "--align_corners",
        action="store_true",
        # type=bool,
        # default=False,
        help="Align corners is true for GPU models and false for mobile models. Or maybe not",
    )
    return parser


# def set_apex_params(local_rank):
#    """
#    Setting distributed parameters for Apex
#    """
#    if "WORLD_SIZE" in os.environ:
#        world_size = int(os.environ["WORLD_SIZE"])
#        global_rank = int(os.environ["RANK"])
#
#    print("GPU {} has Rank {}".format(local_rank, global_rank))
#    torch.cuda.set_device(local_rank)
#    torch.distributed.init_process_group(backend="nccl", init_method="env://")
#    return world_size, global_rank


def evaluate_cityscapes_validation_acc(
    model_name,
    model_weights,
    gpu_id,
    batch_size,
    fp16,
    align_corners,
    num_workers=4,
):
    """
    Inference over dataloader on network
    """
    val_loader = return_dataloader(num_workers, batch_size)

    net = model_entrypoint(model_name)()
    if model_weights:
        net.load_state_dict(
            torch.load(model_weights, map_location=lambda storage, loc: storage)
        )
    if fp16:
        print("Running inference in half precision")

    len_dataset = len(val_loader)
    net.eval()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):
        # Run network
        _iou_acc = eval_minibatch(data, net, True, gpu_id, fp16, align_corners)
        iou_acc += _iou_acc
        if val_idx + 1 < len_dataset:
            printProgressBar(val_idx + 1, len_dataset, "Progress")

    eval_metrics(iou_acc, net)


def main():
    # parse command line
    torch.manual_seed(1234)
    parser = opts_parser()
    args = parser.parse_args()

    # Run inference
    evaluate_cityscapes_validation_acc(**vars(args))


if __name__ == "__main__":
    main()
