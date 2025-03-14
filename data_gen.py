"""
Author: Guoxin Wang
Date: 2024-11-28 12:35:49
LastEditors: Guoxin Wang
LastEditTime: 2025-03-11 11:14:49
FilePath: /DMMECG/data_gen.py
Description:

Copyright (c) 2025 by Guoxin Wang, All Rights Reserved.
"""

import argparse
import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch

from utils.data_utils import ECG_Beat_AF, ECG_Beat_ID

parser = argparse.ArgumentParser(description="Data Generation")
parser.add_argument(
    "--task",
    required=True,
    type=str,
    choices=["af_beat", "id_beat"],
    help="target task",
)
parser.add_argument(
    "--data_path",
    default="./src_data",
    type=str,
    help="path of original dataset",
)
parser.add_argument(
    "--prefix",
    default="",
    type=str,
    help="prefix of datapath",
)
parser.add_argument(
    "--output_dir",
    default="./datasets",
    type=str,
    help="path where to save",
)
parser.add_argument(
    "--width",
    default=240,
    type=int,
    help="half width",
)
parser.add_argument(
    "--channel_names",
    default=None,
    action="append",
    help="list of channels to use",
)
parser.add_argument(
    "--num_class",
    default=4,
    type=int,
    help="number of classes",
)
parser.add_argument("--expansion", default=1, type=int, help="expansion factor")
parser.add_argument(
    "--inter", default=False, action="store_true", help="inter-patient for mitdb"
)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def af_beat() -> None:
    # Uses DS1 DS2 as training set and testing set for MITDB
    if args.inter:
        DS1 = [
            101,
            106,
            108,
            109,
            112,
            114,
            115,
            116,
            118,
            119,
            122,
            124,
            201,
            203,
            205,
            207,
            208,
            209,
            215,
            220,
            223,
            230,
        ]
        DS2 = [
            100,
            103,
            105,
            111,
            113,
            117,
            121,
            123,
            200,
            202,
            210,
            212,
            213,
            214,
            219,
            221,
            222,
            228,
            231,
            232,
            233,
            234,
        ]
        train_files = np.array(
            [
                os.path.join(
                    args.data_path,
                    str(file_name),
                )
                for file_name in DS1
            ]
        )
        test_files = np.array(
            [
                os.path.join(
                    args.data_path,
                    str(file_name),
                )
                for file_name in DS2
            ]
        )
        dataset = ECG_Beat_AF(
            files=train_files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        train_size = int(0.9 * len(dataset))
        train_set = copy.deepcopy(dataset)
        valid_set = copy.deepcopy(dataset)
        train_set.signals = dataset.signals[:train_size]
        train_set.labels = dataset.labels[:train_size]
        valid_set.signals = dataset.signals[train_size:]
        valid_set.labels = dataset.labels[train_size:]
        torch.save(
            train_set,
            "{}/{}_{}_train.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            valid_set,
            "{}/{}_{}_valid.pth".format(args.output_dir, args.task, args.num_class),
        )
        test_set = ECG_Beat_AF(
            files=test_files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        torch.save(
            test_set,
            "{}/{}_{}_test.pth".format(args.output_dir, args.task, args.num_class),
        )
    else:
        files = np.array(
            [
                os.path.join(args.data_path, file_name.split(".")[0])
                for file_name in os.listdir(args.data_path)
                if file_name.endswith(".hea")
            ]
        )
        dataset = ECG_Beat_AF(
            files=files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        train_set = copy.deepcopy(dataset)
        valid_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)
        train_set.signals = dataset.signals[:train_size]
        train_set.labels = dataset.labels[:train_size]
        valid_set.signals = dataset.signals[train_size : train_size + valid_size]
        valid_set.labels = dataset.labels[train_size : train_size + valid_size]
        test_set.signals = dataset.signals[train_size + valid_size :]
        test_set.labels = dataset.labels[train_size + valid_size :]
        torch.save(
            train_set,
            "{}/{}_{}_train.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            valid_set,
            "{}/{}_{}_valid.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            test_set,
            "{}/{}_{}_test.pth".format(args.output_dir, args.task, args.num_class),
        )


def id_beat() -> None:
    folders = [
        os.path.join(args.data_path, folder)
        for folder in os.listdir(args.data_path)
        if folder.startswith(args.prefix)
    ]
    dataset = ECG_Beat_ID(
        folders=folders,
        width=args.width,
        channel_names=args.channel_names,
        expansion=args.expansion,
    )
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    train_set = copy.deepcopy(dataset)
    valid_set = copy.deepcopy(dataset)
    test_set = copy.deepcopy(dataset)
    train_set.signals = dataset.signals[:train_size]
    train_set.labels = dataset.labels[:train_size]
    valid_set.signals = dataset.signals[train_size : train_size + valid_size]
    valid_set.labels = dataset.labels[train_size : train_size + valid_size]
    test_set.signals = dataset.signals[train_size + valid_size :]
    test_set.labels = dataset.labels[train_size + valid_size :]
    torch.save(
        train_set,
        "{}/{}_train.pth".format(args.output_dir, args.task),
    )
    torch.save(
        valid_set,
        "{}/{}_valid.pth".format(args.output_dir, args.task),
    )
    torch.save(
        test_set,
        "{}/{}_test.pth".format(args.output_dir, args.task),
    )


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Data Generation for Task {}".format(args.task))
    if args.task == "af_beat":
        af_beat()
    elif args.task == "id_beat":
        id_beat()


if __name__ == "__main__":
    main()
