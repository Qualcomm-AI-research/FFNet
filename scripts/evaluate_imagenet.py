#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

################################################################
# Code in this script adapted from https://github.com/mehtadushy/SelecSLS-Pytorch/
# Originally made available under CC-BY-4.0
################################################################

"""
Script for evaluating accuracy on Imagenet Validation Set.
"""
import os
import logging
import sys
import time
from argparse import ArgumentParser
import importlib

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
from config import imagenet_base_path
from datasets.imagenet.imagenet_data_loader import get_data_loader
from models.model_registry import model_entrypoint


def opts_parser():
    usage = "Configure the dataset using imagenet_data_loader"
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
    parser.add_argument(
        "--imagenet_base_path",
        type=str,
        default=imagenet_base_path,
        metavar="FILE",
        help="Path to ImageNet dataset",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use.")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size of the input"
    )
    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_imagenet_validation_accuracy(
    model_name, model_weights, imagenet_base_path, gpu_id, batch_size
):
    net = model_entrypoint(model_name)()
    if model_weights:
        net.load_state_dict(
            torch.load(model_weights, map_location=lambda storage, loc: storage)
        )

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.half()
    net = net.to(device)
    net.eval()
    _, test_loader = get_data_loader(
        augment=False, batch_size=batch_size, base_path=imagenet_base_path
    )
    with torch.no_grad():
        val1_acc = []
        val5_acc = []
        for x, y in test_loader:
            pred = F.log_softmax(net(x.half().to(device)))
            top1, top5 = accuracy(pred, y.half().to(device), topk=(1, 5))
            val1_acc.append(top1)
            val5_acc.append(top5)
        avg1_acc = float(np.sum(val1_acc)) / len(val1_acc)
        avg5_acc = float(np.sum(val5_acc)) / len(val5_acc)
    print("Top-1 Error: {} Top-5 Error {}".format(avg1_acc, avg5_acc))
    return [avg1_acc, avg5_acc]


def main():
    # parse command line
    torch.manual_seed(1234)
    parser = opts_parser()
    args = parser.parse_args()

    # run
    _ = evaluate_imagenet_validation_accuracy(**vars(args))


if __name__ == "__main__":
    main()
