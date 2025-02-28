"""
Author: Guoxin Wang
Date: 2023-07-01 16:36:58
LastEditors: Guoxin Wang
LastEditTime: 2025-02-28 12:13:56
FilePath: /DMMECG/engine.py
Description: 

Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
"""

import itertools
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
            experts_probs = torch.stack(
                [softmax(expert(samples), dim=1) for expert in experts], dim=1
            )
            weights = router(samples).view(experts_probs.shape)
            loss = 0
            for expert_idx in range(1, len(experts)):
                idx_tensor = torch.tensor(
                    [expert_idx - 1, expert_idx], device=weights.device
                )
                weights_subset = weights.index_select(dim=1, index=idx_tensor)
                experts_probs_subset = experts_probs.index_select(
                    dim=1, index=idx_tensor
                )
                norm_weights = softmax(weights_subset, dim=1)
                weighted_probs = (norm_weights * experts_probs_subset).sum(dim=1)

                targets = targets.view(-1, 1)
                logpt = torch.log(weighted_probs)
                logpt = logpt.gather(1, targets)
                logpt = logpt.view(-1)
                pt = logpt.exp()
                loss_e = -1 * (1 - pt) ** 3 * logpt
                loss += loss_e.sum()

                # loss += nll_loss(torch.log(weighted_probs), targets)

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
            experts_probs = torch.stack(
                [softmax(expert(samples), dim=1) for expert in experts], dim=1
            )
            weights = router(samples).view(experts_probs.shape)
            norm_weights = softmax(weights, dim=1)
            weighted_probs = (experts_probs * (norm_weights)).sum(dim=1)

            loss = nll_loss(torch.log(weighted_probs), targets)

        acc1, acc3 = accuracy(weighted_probs, targets, topk=(1, 3))

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
                            expert_logits.size(),
                            dtype=expert_logits.dtype,
                            device=expert_logits.device,
                        )
                    mask = (
                        torch.max(softmax(expert_logits, dim=1), dim=1).values
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
def evaluate_a(data_loader, thresholds, experts, device):
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
            weights = torch.tensor(
                [0.9460226704998516, 0.9515190567062376, 0.9545390488543681]
            )
            weights = softmax(weights / 0.01, dim=0)
            outputs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == 0:
                    expert_logits = expert(samples[inference_indices])
                    outputs = torch.zeros(
                        expert_logits.size(),
                        dtype=expert_logits.dtype,
                        device=expert_logits.device,
                    )
                    weighted_logits = expert_logits * weights[expert_idx]
                    mask = (
                        torch.max(softmax(weighted_logits, dim=1), dim=1).values
                        > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_logits[mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )
                elif expert_idx == len(experts) - 1:
                    expert_logits = expert(samples[inference_indices])
                    weighted_logits = expert_logits * weights[expert_idx]
                    outputs[inference_indices] = weighted_logits
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_logits = expert(samples[inference_indices])
                    weighted_logits = expert_logits * weights[expert_idx]
                    mask = (
                        torch.max(softmax(weighted_logits, dim=1), dim=1).values
                        > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_logits[mask]
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
def evaluate_b(data_loader, thresholds, experts, device):
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
            weights = torch.tensor([0.1970, 0.3413, 0.4617])
            outputs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == 0:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    outputs = torch.zeros(
                        expert_probs.size(),
                        dtype=expert_probs.dtype,
                        device=expert_probs.device,
                    )
                    weighted_probs = expert_probs * weights[expert_idx]
                    mask = (
                        torch.max(weighted_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_probs[mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )
                elif expert_idx == len(experts) - 1:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    weighted_probs = expert_probs * weights[expert_idx]
                    outputs[inference_indices] = weighted_probs
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    weighted_probs = expert_probs * weights[expert_idx]
                    mask = (
                        torch.max(weighted_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_probs[mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )

            loss = nll_loss(torch.log(outputs), targets)

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
def evaluate_c(data_loader, thresholds, experts, device):
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
            weights = torch.tensor(
                [94.60226704998516, 95.15190567062376, 95.45390488543681]
            )
            outputs = None
            prev_probs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == 0:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    outputs = torch.zeros(
                        expert_probs.size(),
                        dtype=expert_probs.dtype,
                        device=expert_probs.device,
                    )
                    mask = (
                        torch.max(expert_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = expert_probs[mask]
                    prev_probs = expert_probs[~mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )
                elif expert_idx == len(experts) - 1:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    idx_tensor = torch.tensor(
                        [expert_idx, expert_idx - 1], device=weights.device
                    )
                    weights_subset = weights.index_select(dim=0, index=idx_tensor)
                    norm_weights = softmax(weights_subset, dim=0)
                    weighted_probs = (
                        norm_weights[0] * prev_probs + norm_weights[1] * expert_probs
                    )
                    outputs[inference_indices] = weighted_probs
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    idx_tensor = torch.tensor(
                        [expert_idx, expert_idx - 1], device=weights.device
                    )
                    weights_subset = weights.index_select(dim=0, index=idx_tensor)
                    norm_weights = softmax(weights_subset, dim=0)
                    weighted_probs = (
                        norm_weights[0] * prev_probs + norm_weights[1] * expert_probs
                    )
                    mask = (
                        torch.max(weighted_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_probs[mask]
                    prev_probs = expert_probs[~mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )

            loss = nll_loss(torch.log(outputs), targets)

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
def evaluate_d(data_loader, thresholds, experts, device):
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
            weights = torch.tensor(
                [94.60226704998516, 95.15190567062376, 95.45390488543681]
            )
            outputs = None
            prev_logits = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == 0:
                    expert_logits = expert(samples[inference_indices])
                    outputs = torch.zeros(
                        expert_logits.size(),
                        dtype=expert_logits.dtype,
                        device=expert_logits.device,
                    )
                    mask = (
                        torch.max(softmax(expert_logits, dim=1), dim=1).values
                        > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = expert_logits[mask]
                    prev_logits = expert_logits[~mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )
                elif expert_idx == len(experts) - 1:
                    expert_logits = expert(samples[inference_indices])
                    idx_tensor = torch.tensor(
                        [expert_idx, expert_idx - 1], device=weights.device
                    )
                    weights_subset = weights.index_select(dim=0, index=idx_tensor)
                    norm_weights = softmax(weights_subset, dim=0)
                    weighted_logits = (
                        norm_weights[0] * prev_logits + norm_weights[1] * expert_logits
                    )
                    outputs[inference_indices] = weighted_logits
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_logits = expert(samples[inference_indices])
                    idx_tensor = torch.tensor(
                        [expert_idx, expert_idx - 1], device=weights.device
                    )
                    weights_subset = weights.index_select(dim=0, index=idx_tensor)
                    norm_weights = softmax(weights_subset, dim=0)
                    weighted_logits = (
                        norm_weights[0] * prev_logits + norm_weights[1] * expert_logits
                    )
                    mask = (
                        torch.max(softmax(weighted_logits, dim=1), dim=1).values
                        > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_logits[mask]
                    prev_logits = expert_logits[~mask]
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
def evaluate_e(data_loader, router, thresholds, experts, device):
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
            weights = router(samples).view(batch_size, len(experts), -1)
            outputs = None
            prev_probs = None
            indices = torch.arange(batch_size, device=samples.device)
            inference_indices = indices
            for expert_idx, expert in enumerate(experts):
                if inference_indices.size(0) == 0:
                    for left_idx in range(expert_idx, len(experts)):
                        metric_logger.meters[f"dist{left_idx}"].update(0)
                    break
                if expert_idx == 0:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    outputs = torch.zeros(
                        expert_probs.size(),
                        dtype=expert_probs.dtype,
                        device=samples.device,
                    )
                    mask = (
                        torch.max(expert_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = expert_probs[mask]
                    prev_probs = expert_probs[~mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )
                elif expert_idx == len(experts) - 1:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    idx_tensor = torch.tensor(
                        [expert_idx - 1, expert_idx], device=weights.device
                    )
                    weights_subset = weights[inference_indices].index_select(
                        dim=1, index=idx_tensor
                    )
                    expert_probs_subset = torch.stack([prev_probs, expert_probs], dim=1)
                    norm_weights = softmax(weights_subset, dim=1)
                    weighted_probs = (norm_weights * expert_probs_subset).sum(dim=1)
                    outputs[inference_indices] = weighted_probs
                    metric_logger.meters[f"dist{len(experts)-1}"].update(
                        inference_indices.size(0)
                    )
                else:
                    expert_probs = softmax(expert(samples[inference_indices]), dim=1)
                    idx_tensor = torch.tensor(
                        [expert_idx - 1, expert_idx], device=weights.device
                    )
                    weights_subset = weights[inference_indices].index_select(
                        dim=1, index=idx_tensor
                    )
                    expert_probs_subset = torch.stack([prev_probs, expert_probs], dim=1)
                    norm_weights = softmax(weights_subset, dim=1)
                    weighted_probs = (norm_weights * expert_probs_subset).sum(dim=1)
                    mask = (
                        torch.max(weighted_probs, dim=1).values > thresholds[expert_idx]
                    )
                    outputs[inference_indices[mask]] = weighted_probs[mask]
                    prev_probs = expert_probs[~mask]
                    inference_indices = inference_indices[~mask]
                    metric_logger.meters[f"dist{expert_idx}"].update(
                        torch.sum(mask).item()
                    )

            loss = nll_loss(torch.log(outputs), targets)

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
