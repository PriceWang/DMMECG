"""
Author: Guoxin Wang
Date: 2023-07-01 16:36:58
LastEditors: Guoxin Wang
LastEditTime: 2025-02-18 15:28:35
FilePath: /DMMECG/engine.py
Description: 

Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
"""

import math
import sys
from typing import Iterable

import torch
from timm.utils import accuracy
from torch.nn.functional import cross_entropy, nll_loss, softmax

import utils.lr_sched as lr_sched
import utils.misc as misc


def train_one_epoch(
    router: torch.nn.Module,
    experts: list,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    router.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(1)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device.type):
            experts_logits = torch.stack([expert(samples) for expert in experts], dim=1)
            weighted_logits = router(experts_logits)
            loss = cross_entropy(weighted_logits, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=router.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train/lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid(data_loader, router, experts, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    router.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[-1]
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(1)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device.type):
            experts_logits = torch.stack([expert(samples) for expert in experts], dim=1)
            weighted_logits = router(experts_logits)
            loss = cross_entropy(weighted_logits, targets)

        acc1, acc3 = accuracy(weighted_logits, targets, topk=(1, 3))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc3"].update(acc3.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_lw(data_loader, router, thresholds, experts, device):
    assert len(thresholds) == len(experts) - 1
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[-1]
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(1)
        targets = targets.to(device, non_blocking=True)

        batch_size = samples.size(0)
        # compute output
        with torch.amp.autocast(device.type):
            outputs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == len(experts) - 1:
                    expert_logits = expert(samples[inference_indices])
                    outputs[inference_indices] = expert_logits
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_logits = expert(samples[inference_indices])
                    weighted_logits = router.router[expert_idx](expert_logits)
                    if expert_idx == 0:
                        outputs = torch.zeros(
                            batch_size,
                            weighted_logits.size(1),
                            dtype=expert_logits.dtype,
                            device=samples.device,
                        )
                    mask = (
                        torch.max(weighted_logits, dim=1).values
                        > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = expert_logits[mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )

            loss = cross_entropy(outputs, targets)

        acc1, acc3 = accuracy(outputs, targets, topk=(1, 3))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc3"].update(acc3.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss
        )
    )

    return {
        k: meter.total if k.startswith("dist") else meter.global_avg
        for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluate(data_loader, thresholds, experts, device):
    assert len(thresholds) == len(experts) - 1
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[-1]
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(1)
        targets = targets.to(device, non_blocking=True)

        batch_size = samples.size(0)
        # compute output
        with torch.amp.autocast(device.type):
            outputs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == len(experts) - 1:
                    expert_logits = expert(samples[inference_indices])
                    outputs[inference_indices] = expert_logits
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_logits = expert(samples[inference_indices])
                    if expert_idx == 0:
                        outputs = torch.zeros(
                            batch_size,
                            expert_logits.size(1),
                            dtype=expert_logits.dtype,
                            device=samples.device,
                        )
                    mask = (
                        torch.max(expert_logits, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = expert_logits[mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )

            loss = cross_entropy(outputs, targets)

        acc1, acc3 = accuracy(outputs, targets, topk=(1, 3))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc3"].update(acc3.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss
        )
    )

    return {
        k: meter.total if k.startswith("dist") else meter.global_avg
        for k, meter in metric_logger.meters.items()
    }
