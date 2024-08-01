import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import wandb
from model.model import make_model
from model.Embedding import PositionalEncoding, Embeddings
from loguru import logger
import torch
from utils.util import seed_everything, EarlyStopping, torch_distributed_zero_first
from data.dataset import MyDataset, load_data
from sklearn.model_selection import train_test_split
from torch import nn, optim
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm, trange
from flatten_dict import flatten
import seaborn as sns
import pandas as pd
import math
from torch.optim.lr_scheduler import LambdaLR
import sys
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.append("./model")


# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#         torch.uint8
#     )
#     return subsequent_mask == 0


def do_evaluate(yaml_args, model, data_loader, criterion):

    num_processes = yaml_args.world_size
    val_loss = 0
    nb_eval_steps = 0
    losses = None

    for src_input, tgt_input in tqdm(data_loader, desc="Evalueating"):
        model.eval()

        with torch.no_grad():
            src = src_input["input_ids"]
            src_attmask = src_input["src_mask"].cuda(non_blocking=True)
            tgt = tgt_input["input_ids"]
            tgt_attmask = tgt_input["tgt_mask"].cuda(non_blocking=True)
            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)

            output = model(src, tgt[:, :-1], src_attmask, tgt_attmask)

            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            loss = loss.mean().detach()

            if loss.ndim == 0:
                loss = loss.clone()[None]

            if not loss.is_contiguous():
                loss = loss.contiguous()

            output_tensors = [torch.empty_like(loss) for _ in range(num_processes)]
            reduce_losses = torch.cat(output_tensors, dim=0)
            if command_args.world_size > 1:
                torch.distributed.all_gather(output_tensors, loss)
            losses = reduce_losses if losses is None else torch.cat((losses, reduce_losses), dim=0)
        nb_eval_steps += 1

    return losses.mean()


def do_train(yaml_args, command_args):

    # 1. 设置随机种子
    seed_everything(yaml_args.seed + command_args.local_rank)

    # 2. 处理数据
    with torch_distributed_zero_first(command_args.local_rank):

        data = load_data([yaml_args.Data.in_path, yaml_args.Data.out_path])
        if yaml_args.debug:
            data = data[:1000]

        data_train, data_val = train_test_split(data, test_size=yaml_args.Data.val_ratio, random_state=yaml_args.seed, shuffle=True)

        train_dataset = MyDataset(data_train)
        val_dataset = MyDataset(data_val)

        logger.info(f"Build train dataset with {len(train_dataset)} samples and val dataset with {len(val_dataset)} samples")

    train_loader = train_dataset.get_loader(
        yaml_args.Train.per_gpu_train_batch_size,
        shuffle=True,
        sampler=True if command_args.local_rank != -1 else False,
        num_workers=yaml_args.Data.num_workers,
    )
    val_loader = val_dataset.get_loader(
        yaml_args.Train.per_gpu_train_batch_size,
        shuffle=False,
        sampler=True if command_args.local_rank != -1 else False,
        num_workers=yaml_args.Data.num_workers,
    )
    if command_args.local_rank in [-1, 0]:
        logger.info(f"Build train dataloader with {len(train_loader)} batches and val dataloader with {len(val_loader)} batches")

    # 3.model
    model = make_model()
    model = model.cuda()

    # 4.optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay))], "weight_decay": yaml_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=yaml_args.Optim.learning_rate,
        eps=yaml_args.Optim.adam_epsilon,
    )

    # 5.Scheduler
    lr_max, lr_min = yaml_args.Scheduler.lr_max, yaml_args.Scheduler.lr_min
    T_max = yaml_args.Train.num_epochs * len(train_loader)
    warm_up_iter = int(T_max * yaml_args.Scheduler.warmup_ratio)

    def WarmupExponentialLR(cur_iter):
        gamma = math.exp(math.log(lr_min / lr_max) / (T_max - warm_up_iter))
        if cur_iter < warm_up_iter:
            return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
        else:
            return lr_max * gamma ** (cur_iter - warm_up_iter)

    scheduler = LambdaLR(optimizer, lr_lambda=WarmupExponentialLR)

    # 6.resume
    start_epoch = 0
    if yaml_args.resume:
        checkpoint = torch.load(yaml_args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model-state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

        del checkpoint

    if command_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[command_args.local_rank], output_device=command_args.lock_rank)

    # 7.loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean").cuda()

    # gc.collect()
    # torch.cuda.empty_cache()

    # 8.early_stop
    early_stopping = None
    if yaml_args.EarlyStop.early_stop:
        if command_args.local_rank in [-1, 0]:
            logger.info("Use EarlyStop")
        early_stopping = EarlyStopping(patience=yaml_args.EarlyStop.patience, delta=yaml_args.EarlyStop.delta)

    # t_total = len(train_dataloader) // yaml_args.gradient_accumulation_steps * yaml_args.Train.num_epochs

    # 9. Train !!
    global_step = 0
    tr_loss = 0.0
    scheduler.last_epoch = start_epoch - 1

    model.zero_grad()

    train_iterator = trange(start_epoch, int(yaml_args.Train.num_epochs), desc="Epoch", disable=command_args.local_rank not in [-1, 0])

    for epoch in train_iterator:

        if command_args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(train_loader, desc="Iteration", disable=command_args.local_rank not in [-1, 0])
        for step, (src_input, tgt_input) in enumerate(epoch_iterator):
            model.train()
            global_step += 1

            src = src_input["input_ids"]
            src_attmask = src_input["src_mask"].cuda(non_blocking=True)
            tgt = tgt_input["input_ids"]
            tgt_attmask = tgt_input["tgt_mask"].cuda(non_blocking=True)
            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)

            output = model(src, tgt[:, :-1], src_attmask, tgt_attmask)
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt)

            # 梯度累计
            if yaml_args.gradient_accumulation_steps > 1:
                loss = loss / yaml_args.gradient_accumulation_steps

            loss.backward()
            if command_args.local_rank in [-1, 0]:
                step_metrci = {"loss": loss}
                wandb.log(step_metrci)
            tr_loss += loss.item()

            if (global_step + 1) % yaml_args.gradient_accumulation_steps == 0:
                if yaml_args.max_grad_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), yaml_args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if command_args.local_rank in [-1, 0]:
                    scheduler_last_lr = scheduler.get_last_lr()[0]
                    wandb.log({"scheduler": scheduler_last_lr})

                # 9.1 Do evalute during Train
                if yaml_args.evaluate_during_training and yaml_args.evaluate_steps > 0 and (global_step + 1) % yaml_args.evaluate_steps == 0:
                    eval_loss = do_evaluate(command_args, model, val_loader, criterion)
                    eval_loss_dict = {"eval loss": eval_loss}
                    wandb.log(eval_loss_dict)
                    stop = early_stopping(eval_loss)
                    if stop:
                        return

                if command_args.local_rank in [-1, 0] and (yaml_args.save_steps > 0 and (global_step + 1) % yaml_args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(yaml_args.Save.model_save_dir, "checkpoint-{}.bin".format(global_step))
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    checkpoint = {
                        "model_state": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(checkpoint, output_dir)


def main(yaml_args, command_args):

    logger.add(yaml_args.Log.path)
    wandb.init(project="couplet", name="demo", config=flatten(yaml_args, reducer="dot"))

    if command_args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # command_args.n_gpu = 1
    else:
        assert torch.cuda.device_count() > command_args.local_rank
        torch.cuda.set_device(command_args.local_rank)
        device = torch.device("cuda", command_args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    wandb.alert(title="Start", text=f"training")

    do_train(yaml_args, command_args)

    wandb.finish()


def get_args_from_yaml(yaml_path):
    conf = OmegaConf.load(yaml_path)

    return conf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="world_size")
    command_args = parser.parse_args()

    yaml_args = get_args_from_yaml("para.yaml")

    main(yaml_args, command_args)
