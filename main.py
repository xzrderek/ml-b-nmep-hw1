import argparse
import datetime
import json
import os
import shutil
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from timm.utils import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset  # For custom datasets
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_str

from config import get_config
from data import build_loader
from data.datasets import CIFAR10Dataset
from models import build_model
from optimizer import build_optimizer
from utils import create_logger, load_checkpoint, save_checkpoint


def parse_option():
    parser = argparse.ArgumentParser("Vision model training and evaluation script", add_help=False)
    # saving visualzations of dataset
    # cmd python main.py --cfg configs/resnet18_base.yaml --vis-dataset 10
    parser.add_argument("--vis-dataset", type=int, help="number of images to save to disk from dataset", default=0)
    parser.add_argument("--save", type=int, help="save as csv", default=0)
    
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    # parser.add_argument(
    #     "--output",
    #     default="output",
    #     type=str,
    #     metavar="PATH",
    #     help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    # )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(
        config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config)
    # logger.info(str(model))

    model = model.to(device)

    # param and flop counts
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    toy_input = torch.rand(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).to(device) # for measuring flops
    flops = FlopCountAnalysis(model, toy_input)
    del toy_input

    # print("Model = %s" % str(model_without_ddp))
    n_flops = flops.total()
    logger.info(flop_count_str(flops))
    logger.info('number of params: {} M'.format(n_parameters / 1e6))
    logger.info(f'flops: {n_flops/1e6} MFLOPS')

    # Keep it simple with basic epoch scheduler
    optimizer = build_optimizer(config, model)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS)

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="nmep resnet 50-transform graph",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.0003,
        "architecture": "ResNet",
        "dataset": "CIFAR-100",
        "epochs": 50,
        }
    )
    
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        batch_time = time.time()
        train_acc1, train_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch)
        logger.info(f" * Train Acc {train_acc1:.3f} Train Loss {train_loss:.3f}")
        logger.info(f"Accuracy of the network on the {len(dataset_train)} train images: {train_acc1:.1f}%")

        # train_acc1, _ = validate(config, data_loader_train, model)
        val_acc1, val_loss = validate(config, data_loader_val, model)
        logger.info(f" * Val Acc {val_acc1:.3f} Val Loss {val_loss:.3f}")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {val_acc1:.1f}%")

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        max_accuracy = max(max_accuracy, val_acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%\n")
        lr_scheduler.step()

        log_stats = {"epoch": epoch, "n_params": n_parameters, "n_flops": n_flops,
                     "train_acc": train_acc1, "train_loss": train_loss, 
                     "val_acc": val_acc1, "val_loss": val_loss}
        batch_t = 128 / (time.time() - batch_time)
        wandb.log({"val_acc": val_acc1, "val_loss": val_loss, "train_acc": train_acc1, "train_loss": train_loss, "_throughput_": batch_t})
        with open(
                os.path.join(config.OUTPUT, "metrics.json"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    logger.info("Start testing")
    preds = evaluate(config, data_loader_test, model)
    np.save(os.path.join(config.OUTPUT, "preds.npy"), preds)
    # TODO save predictions to csv in kaggle format


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(tqdm(data_loader, leave=False)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        (acc1,) = accuracy(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    lr = optimizer.param_groups[0]["lr"]
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"Train: [{epoch}/{config.TRAIN.EPOCHS}]\t"
        f"lr {lr:.6f}\t"
        f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
        f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
        f"Mem {memory_used:.0f}MB"
    )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        (acc1,) = accuracy(output, target)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"Validate: \t"
        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
        f"Mem {memory_used:.0f}MB"
    )
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def evaluate(config, data_loader, model):
    model.eval()
    preds = []
    for idx, (images, _) in enumerate(tqdm(data_loader)):
        images = images.cuda(non_blocking=True)
        output = model(images)
        preds.append(output.cpu().numpy())
    preds = np.concatenate(preds)
    return preds


if __name__ == "__main__":
    args, config = parse_option()
    
    # select vis-dataset
    if vars(args)["vis_dataset"] != 0:
        dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(
            config
        )
        
        for i in range(vars(args)["vis_dataset"]):
            im = CIFAR10Dataset.__getitem__(dataset_train, random.randint(0, CIFAR10Dataset.__len__(dataset_train)))
            label = im[1]
            image = im[0]
            image.reshape(3,32,32).permute(1, 2, 0)
            image_path = str(label) + "no" + str(i) + "image.png"
            # save image
            tensor  = image.cpu()
            save_image(tensor, image_path)
            # show image
            image = mpimg.imread(image_path)
            plt.title = image_path
            plt.imshow(image)
            plt.show()
    elif vars(args)["save"] != 0:
        import pandas as pd
        import numpy as np

        PATH = 'output/resnet18/preds.npy'

        data = np.load(PATH)
        data = np.argmax(data, axis=1)

        df = pd.DataFrame(data)
        df.index += 1
        df.to_csv('submission.csv', header=['Category'], index_label='Id')
    else:
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
