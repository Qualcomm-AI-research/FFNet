#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

################################################################
# Code in this script adapted from https://github.com/mehtadushy/SelecSLS-Pytorch/
# Originally made available under CC-BY-4.0
################################################################


"""
Script for timing models in eval mode and torchscript eval modes.
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

from models.model_registry import model_entrypoint


def opts_parser():
    usage = "Pass the model name, weights, input size, gpu id"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        "--num_iter", type=int, default=50, help="Number of iterations to average over."
    )
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
        "--input_size",
        nargs=2,
        type=int,
        default=[1024, 2048],
        help="Input image size.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of the input"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use.")
    return parser


def measure_cpu(model, x, fp16=False):
    # synchronize gpu time and measure fp
    if fp16:
        x = x.half()
        model = model.half()
    else:
        x = x.float()
        model = model.float()
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        y_pred = model(x)
        elapsed_fp_nograd = time.time() - t0
    return elapsed_fp_nograd


def measure_gpu(model, x, fp16=False):
    # synchronize gpu time and measure fp
    if fp16:
        x = x.half()
        model = model.half()
    else:
        x = x.float()
        model = model.float()
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        y_pred = model(x)
        torch.cuda.synchronize()
        elapsed_fp_nograd = time.time() - t0
    return elapsed_fp_nograd


def benchmark(model_name, gpu_id, num_iter, model_weights, input_size, batch_size):
    # Import the model module
    net = model_entrypoint(model_name)()
    if model_weights:
        net.load_state_dict(
            torch.load(model_weights, map_location=lambda storage, loc: storage)
        )

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print("\nEvaluating on GPU {}".format(device))

    print(f"\nGPU, Batch Size: {batch_size}")
    x = torch.randn(batch_size, 3, input_size[0], input_size[1])
    # Warm up
    for i in range(10):
        _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t = measure_gpu(net, x.to(device))
        fp.append(t)
    fp32_timing = np.mean(np.asarray(fp) * 1000)
    print(f"FP32 Model FP: {fp32_timing} ms")
    # Warm up
    for i in range(10):
        _ = measure_gpu(net, x.to(device), fp16=True)
    fp = []
    for i in range(num_iter):
        t = measure_gpu(net, x.to(device), fp16=True)
        fp.append(t)
    fp16_timing = np.mean(np.asarray(fp) * 1000)
    print(f"FP16 Model FP: {fp16_timing} ms")
    return fp32_timing, fp16_timing

    # jit_net = torch.jit.trace(net, x.to(device))
    # for i in range(10):
    #    _ = measure_gpu(jit_net, x.to(device))
    # fp = []
    # for i in range(num_iter):
    #    t = measure_gpu(jit_net, x.to(device))
    #    fp.append(t)
    # print("JIT FP: " + str(np.mean(np.asarray(fp) * 1000)) + "ms")


def main():
    # parse command line
    torch.manual_seed(1234)
    parser = opts_parser()
    args = parser.parse_args()

    # run
    _ = benchmark(**vars(args))


if __name__ == "__main__":
    main()
