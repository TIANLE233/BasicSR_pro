import os
import numpy as np
import time, datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
from thop import profile
from gpu_energy_estimation import GPUEnergyEvaluator
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed


def main(model, gpu=0):
    cudnn.benchmark = True
    cudnn.enabled = True

    # gpu = args.gpu
    device = torch.device(gpu)
    # model = eval(arch)
    print(model)
    model = model.to(device)

    input_image_size = 224
    input_image = torch.randn(1, 4, input_image_size, input_image_size).to(device)
    flops, params = profile(model, inputs=(input_image,))
    print('Params: %.2f M' % (params / 1e6))
    print('Flops: %.2f G' % (flops / 1e9))

    # model = eval(args.arch)(sparsity=sparsity).cuda()

    # load training data
    print('==> Preparing data..')
    batch_size = 64

    model.eval()
    times = []

    input = torch.randn(batch_size, 4, input_image_size, input_image_size, dtype=torch.float32)
    input = input.to(device)
    with torch.no_grad():
        for i in range(40):
            output = model(input)
            torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    evaluator = GPUEnergyEvaluator(gpuid=gpu)

    model.eval()
    input = input.to(device)
    evaluator.start()
    with torch.no_grad():
        for i in range(100):
            start_evt.record()
            output = model.forward(input)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_time = start_evt.elapsed_time(end_evt)

            times.append(elapsed_time)
    energy_used = evaluator.end()

    print("Energy used (J/image)", energy_used / (100 * batch_size))
    print("FPS:", batch_size * 1e+3 / np.mean(times))

