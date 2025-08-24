import logging
import argparse
import random

import torch
import numpy as np

from Process.Train import train
from Process.Test import test
from Model import get_model

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--source_list", nargs='+', help="Path of the training data.")
    parser.add_argument("--target_list", help="Path of the test data.")
    parser.add_argument("--dataset_path", type=str, help="The base path of the dataset.")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes in the dataset.")
    parser.add_argument("--backbone_type", choices=["ViT-B_16", "DINOv2"],
                        default="DINOv2", help="Which variant to use.")
    parser.add_argument("--backbone_size", choices=["base", "large", "huge"])
    parser.add_argument("--bottleneck_size", choices=["small", "base", "large", "huge"])

    # TODO: Change pretrained path
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")
    parser.add_argument("--source_only", default=False, action="store_true",
                        help="Train without SDAL.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_rounds", default=20, type=int,
                        help="Total number of rounds transfer data between clients and server.")
    parser.add_argument("--num_steps_clients", default=20, type=int,
                        help="Number of epochs for each client to train its model for each round.")
    parser.add_argument("--main_steps", default=20, type=int)

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps_clients", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--warmup_steps_main", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--gpu_ids", nargs="+", default=[0], type=int,
                        help="ID of GPUs")
    parser.add_argument("--exp_num", type=int, required=True)
    parser.add_argument("--train_from_scratch", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logging.info(f"Adapting {args.source_list} to {args.target_list}")
    logging.info(f"GPUs: {args.gpu_ids}")

    set_seed(args)

    logger.info(f"Training each client for {args.num_steps_clients} steps.")
    logger.info(f"Training from scratch? {args.train_from_scratch}")

    args.logger = logger

    if args.is_test:
        args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        model = get_model(args)
        test(args, model)
    else:
        train(args)


if __name__ == "__main__":
    main()
