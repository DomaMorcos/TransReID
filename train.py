import os
import argparse
import logging
import torch
from config import cfg
try:
    from datasets.make_dataloader import make_dataloader
    print("Imported make_dataloader successfully:", make_dataloader)
except ImportError as e:
    print(f"Failed to import make_dataloader: {e}")
    raise
from model import make_model
from solver import make_optimizer
from loss import make_loss
from processor import do_train
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="TransReID Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger("transreid")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Saving model in the path: {output_dir}")
    logger.info(args)

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    # Data
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.cuda()

    # Loss
    loss_func = make_loss(cfg, num_classes=num_classes)

    # Optimizer
    optimizer = make_optimizer(cfg, model)

    # Training
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        num_query,
    )

if __name__ == "__main__":
    main()