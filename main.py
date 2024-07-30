import wandb
from model.model import make_model
from model.Embedding import PositionalEncoding, Embeddings
from loguru import logger
import torch
from utils.util import seed_everything, EarlyStopping
from data.dataset import MyDataset, load_data
from sklearn.model_selection import train_test_split
from torch import nn, optim
import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
import yaml
import argparse
import seaborn as sns
import pandas as pd
import math
from torch.optim.lr_scheduler import LambdaLR
import sys
import torch.distributed as dist

sys.path.append('./model')


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def train_one_epoch(config, model, train_loader, optimizer, criterion, scheduler):
    model.train()
    train_loss = 0
    for src_input, tgt_input in tqdm(train_loader, desc=f"Epoch", leave=False, position=1):
        src = src_input['input_ids']
        src_attmask = src_input['src_mask'].cuda(non_blocking=True)

        tgt = tgt_input['input_ids']
        tgt_attmask = tgt_input['tgt_mask'].cuda(non_blocking=True)

        src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
        output = model(src, tgt[:, :-1], src_attmask, tgt_attmask)

        output = output.contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(output, tgt)
        step_metrci = {'loss':loss}
        wandb.log(step_metrci)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return train_loss / len(train_loader)


def evaluate(config, model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src_input, tgt_input in tqdm(val_loader, desc=f"Val", leave=False):
            src = src_input['input_ids']
            src_attmask = src_input['src_mask'].cuda(non_blocking=True)

            tgt = tgt_input['input_ids']
            tgt_attmask = tgt_input['tgt_mask'].cuda(non_blocking=True)

            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
            output = model(src, tgt[:, :-1], src_attmask, tgt_attmask)

            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def train_model(config, model, train_loader, val_loader, optimizer, criterion, scheduler):
    best_loss = float("inf")
    history = []
    model_path = os.path.join(config.Save.model_save_dir, f"model_best.pth")

    if config.EarlyStop.early_stop:
        logger.info('Use EarlyStop')
        early_stopping = EarlyStopping(patience=config.EarlyStop.patience, delta=config.EarlyStop.delta)

    for epoch in tqdm(range(1, config.Train.epochs + 1), desc=f"All", position=0):
        train_loss = train_one_epoch(
            config, model, train_loader, optimizer, criterion, scheduler)

        val_loss = evaluate(config, model, val_loader, criterion)

        perplexity = math.exp(val_loss)
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        wandb.log(metrics)
        history.append((epoch, train_loss, val_loss))
        msg = f"Epoch {epoch}/{config.Train.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Perplexity: {perplexity:.4f}"
        logger.info(msg)
        if val_loss < best_loss:
            logger.info(
                f"Val loss decrease from {best_loss:>10.6f} to {val_loss:>10.6f}"
            )
            torch.save(model.state_dict(), model_path)
            best_loss = val_loss
        if config.EarlyStop.early_stop:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    logger.info(f"Save best model with val loss {best_loss:.6f} to {model_path}")

    model_path = os.path.join(config.model_save_dir, f"{model.name}_last.pth")
    
    checkpoint = {'net':model.state_dict(),
                  'optimizer':optimizer.state_dict(),
                  "epoch": epoch,
                  'scheduler':scheduler.state_dict()}
    
    torch.save(checkpoint, model_path)
    logger.info(f"Save last model with val loss {val_loss:.6f} to {model_path}")

    history = pd.DataFrame(
        history, columns=["Epoch", "Train Loss", "Val Loss"]
    ).set_index("Epoch")
    history.plot(
        subplots=True, layout=(1, 2), sharey="row", figsize=(14, 6), marker="o", lw=2
    )
    history_path = os.path.join(config.img_save_dir, "history.png")
    plt.savefig(history_path, dpi=300)
    logger.info(f"Save history to {history_path}")


def main(config):
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(config.local_rank)
    
    wandb.alert(
        title="Start",
        text=f"training"
    )

    seed_everything(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    if config.cuDNN:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    data = load_data([config.Data.in_path, config.Data.out_path])

    if config.debug:
        data = data[:1000]
    logger.info(f"Load {len(data)} couplets")

    data_train, data_val = train_test_split(
        data, test_size=config.Data.val_ratio, random_state=config.seed, shuffle=True
    )

    train_dataset = MyDataset(data_train)
    val_dataset = MyDataset(data_val)
    logger.info(f"Build train dataset with {len(train_dataset)} samples")
    logger.info(f"Build val dataset with {len(val_dataset)} samples")

    train_loader = train_dataset.get_loader(
        config.Train.batch_size, shuffle=True,sampler=True)
    val_loader = val_dataset.get_loader(
        config.Train.batch_size, shuffle=False)
    logger.info(f"Build train dataloader with {len(train_loader)} batches")
    logger.info(f"Build val dataloader with {len(val_loader)} batches")

    model = make_model()
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank])
    logger.info(f"Build model with mdoel")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1,
        betas=(config.Optim.beta1, config.Optim.beta2),
        eps=config.Optim.epsilon,
        weight_decay=config.Optim.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean").cuda()

    # scheduler
    lr_max, lr_min = config.Scheduler.lr_max, config.Scheduler.lr_min
    T_max = config.Train.epochs * len(train_loader)
    warm_up_iter = int(T_max * config.Scheduler.warmup_ratio)

    def WarmupExponentialLR(cur_iter):
        gamma = math.exp(math.log(lr_min / lr_max) / (T_max - warm_up_iter))
        if cur_iter < warm_up_iter:
            return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
        else:
            return lr_max * gamma ** (cur_iter - warm_up_iter)

    logger.info('Use warm up')
    scheduler = LambdaLR(optimizer, lr_lambda=WarmupExponentialLR)

    if config.local_rank == 0:
        df_lr = pd.DataFrame(
            [WarmupExponentialLR(i) for i in range(T_max)],
            columns=["Learning Rate"],
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_lr, linewidth=2)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        lr_img_path = os.path.join(config.Save.img_save_dir, "lr_schedule.png")
        plt.savefig(lr_img_path, dpi=300)
        logger.info(f"Save learning rate schedule to {lr_img_path}")

    gc.collect()
    # torch.cuda.empty_cache()

    train_model(
        config, model, train_loader, val_loader, optimizer, criterion, scheduler
    )


def get_args():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open('para.yaml', 'r') as f:
        config = yaml.safe_load(f)
        new_config = dict2namespace(config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(new_config, key, value)
        
    return new_config


if __name__ == '__main__':
    logger.add('./logger.log')
    config = get_args()
    wandb.init(project='couplet', name='demo',config=config)
    main(config)
    wandb.finish()  
