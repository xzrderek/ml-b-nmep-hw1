import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from config import get_config
import argparse
import numpy as np
import json
import datetime
import os
import time
import shutil
from logger import create_logger
from data import build_loader
from models import build_model

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
    	#TODO: use the usual pytorch implementation of training

def validate(val_loader, model, criterion):
	model.eval()
    for i, (input, target) in enumerate(val_loader):
    	#TODO: implement the validation. Remember this is validation and not training
    	#so some things will be different.

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	#best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    model = build_model(config)
    logger.info(str(model))

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")
    # if hasattr(model, 'flops'):
    #     flops = model.flops()
    #     logger.info(f"number of GFLOPs: {flops / 1e9}")

    # model.cuda()

    # can be simple
    optimizer = build_optimizer(config, model)

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    # if config.THROUGHPUT_MODE:
    #     throughput(data_loader_val, model, logger)
    #     return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]['lr']

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f' * Acc {acc1:.3f} Loss {loss:.3f}')
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))




if __name__ == "__main__":
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # Make output dir 
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
