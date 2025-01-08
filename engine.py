"""
Author: Guoxin Wang
Date: 2023-07-01 16:36:58
LastEditors: Guoxin Wang
LastEditTime: 2025-01-08 14:24:27
FilePath: /DNSECG/engine.py
Description: 

Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
"""

import math
import sys
from typing import Iterable

import torch
from timm.utils import accuracy
from torch.nn.functional import cross_entropy, mse_loss, softmax

import utils.lr_sched as lr_sched
import utils.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    experts: list,
    complexity_dist: torch.Tensor,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train()
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
            batch_size = samples.size(0)

            gate_outputs = model(samples).squeeze(1)

            experts_loss = torch.tensor(
                [cross_entropy(expert(samples), targets) for expert in experts]
            ).to(device)
            experts_dist = softmax(
                experts_loss / torch.max(experts_loss) / args.tau_g,
                dim=0,
            )

            loss_g = mse_loss(gate_outputs, 1 - experts_dist.repeat(batch_size, 1))
            loss_p = mse_loss(gate_outputs, 1 - complexity_dist.repeat(batch_size, 1))
            loss = mse_loss(
                gate_outputs,
                1
                - (
                    (1 - args.lam) * experts_dist.repeat(batch_size, 1)
                    + args.lam * complexity_dist.repeat(batch_size, 1)
                ),
            )
        loss_gate = loss_g.item()
        loss_penalty = loss_p.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_g=loss_gate)
        metric_logger.update(loss_p=loss_penalty)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_gate_reduce = misc.all_reduce_mean(loss_gate)
        loss_penalty_reduce = misc.all_reduce_mean(loss_penalty)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train/loss_gate", loss_gate_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train/loss_penalty", loss_penalty_reduce, epoch_1000x
            )
            log_writer.add_scalar("train/lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, experts, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

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
            gate_outputs = model(samples).squeeze(1)
            gate_dist = softmax(gate_outputs, dim=1)
            expert_idx = torch.argmax(gate_dist, dim=1)

            batch_size = samples.size(0)
            outputs_list = []
            for expert_id in range(len(experts)):
                # mask for selecting samples assigned to this expert
                mask = expert_idx == expert_id
                metric_logger.meters[f"dist{expert_id}"].update(torch.sum(mask).item())
                selected_samples = samples[mask]

                # only forward selected samples
                if selected_samples.size(0) > 0:
                    expert_outputs = experts[expert_id](selected_samples)
                    outputs_list.append((mask, expert_outputs))

            # placeholder for final outputs
            outputs = torch.zeros(
                batch_size,
                outputs_list[0][1].size(1),
                device=samples.device,
                dtype=outputs_list[0][1].dtype,
            )
            for mask, expert_outputs in outputs_list:
                outputs[mask] = expert_outputs

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
