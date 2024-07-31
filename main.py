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

    val_loss = 0
    nb_eval_steps = 0

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

            val_loss += loss.item()
        nb_eval_steps += 1

    return val_loss / nb_eval_steps


def do_train(yaml_args, command_args, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion):

    early_stopping = None
    if yaml_args.EarlyStop.early_stop:
        logger.info("Use EarlyStop")
        early_stopping = EarlyStopping(patience=yaml_args.EarlyStop.patience, delta=yaml_args.EarlyStop.delta)

    t_total = len(train_dataloader) // yaml_args.gradient_accumulation_steps * yaml_args.Train.num_epochs

    global_step = 0.0
    tr_loss = 0.0
    model.zero_grad()

    epoch_start = -1
    if yaml_args.resume:
        epoch_start = yaml_args.epoch_start

    train_iterator = trange(epoch_start + 1, int(yaml_args.Train.num_epochs), desc="Epoch", disable=command_args.local_rank not in [-1, 0])

    for spoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=command_args.local_rank not in [-1, 0])
        for step, (src_input, tgt_input) in enumerate(epoch_iterator):
            model.train()
            train_loss = 0

            src = src_input["input_ids"]
            src_attmask = src_input["src_mask"].cuda(non_blocking=True)
            tgt = tgt_input["input_ids"]
            tgt_attmask = tgt_input["tgt_mask"].cuda(non_blocking=True)
            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)

            output = model(src, tgt[:, :-1], src_attmask, tgt_attmask)
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt)

            if yaml_args.gradient_accumulation_steps > 1:
                loss = loss / yaml_args.gradient_accumulation_steps

            loss.backward()
            step_metrci = {"loss": loss}
            wandb.log(step_metrci)
            tr_loss += loss.item()

            if (step + 1) % yaml_args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), yaml_args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if command_args.local_rank in [-1, 0] and yaml_args.logging_steps > 0 and global_step % yaml_args.logging_steps == 0:
                    # Log metrics
                    if (
                        command_args.local_rank == -1 and yaml_args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        eval_loss = do_evaluate(yaml_args, model, val_dataloader, criterion)
                        scheduler_wandb = scheduler.get_last_lr()[0]
                        eval_loss_dict = {"eval loss": eval_loss, "scheduler": scheduler_wandb}
                        wandb.log(eval_loss_dict)
                        stop = early_stopping(eval_loss)
                        if stop:
                            break
                if command_args.local_rank in [-1, 0] and (yaml_args.save_steps > 0 and global_step % yaml_args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(yaml_args.Save.model_save_dir, "checkpoint-{}.bin".format(global_step))
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    checkpoint = {
                        "model_state": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": spoch,
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(checkpoint, output_dir)


def main(yaml_args, command_args):

    logger.add(yaml_args.Log.path)
    wandb.init(project="couplet", name="demo", config=flatten(yaml_args, reducer="dot"))

    if command_args.local_rank == -1 or yaml_args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        command_args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(command_args.local_rank)
        device = torch.device("cuda", command_args.local_rank)
        dist.init_process_group(backend="nccl")
        command_args.n_gpu = 1
    command_args.device = device

    wandb.alert(title="Start", text=f"training")

    seed_everything(yaml_args, command_args)

    # if yaml_args.cuDNN:
    #     torch.backends.cudnn.enabled = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = True

    if command_args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    data = load_data([yaml_args.Data.in_path, yaml_args.Data.out_path])
    if yaml_args.debug:
        data = data[:1000]

    if command_args.local_rank == 0:
        torch.distributed.barrier()

    model = make_model()
    model = model.cuda()

    data_train, data_val = train_test_split(data, test_size=yaml_args.Data.val_ratio, random_state=yaml_args.seed, shuffle=True)

    # 这个地方还是要改一下
    yaml_args.train_batch_size = yaml_args.Train.per_gpu_train_batch_size * max(command_args.n_gpu, 0)

    train_dataset = MyDataset(data_train)
    val_dataset = MyDataset(data_val)
    logger.info(f"Build train dataset with {len(train_dataset)} samples and val dataset with {len(val_dataset)} samples")

    train_loader = train_dataset.get_loader(
        yaml_args.train_batch_size, shuffle=True, sampler=True if command_args.local_rank != -1 else False, num_workers=yaml_args.Data.num_workers
    )
    val_loader = val_dataset.get_loader(yaml_args.train_batch_size, shuffle=False, num_workers=yaml_args.Data.num_workers)
    logger.info(f"Build train dataloader with {len(train_loader)} batches and val dataloader with {len(val_loader)} batches")

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

    if yaml_args.resume:
        checkpoint = torch.load(yaml_args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model-state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        yaml_args.start = checkpoint["epoch"]
        last_step = yaml_args.epoch_start * (len(train_loader) // yaml_args.gradient_accumulation_steps)

    lr_max, lr_min = yaml_args.Scheduler.lr_max, yaml_args.Scheduler.lr_min
    T_max = yaml_args.Train.num_epochs * len(train_loader)
    warm_up_iter = int(T_max * yaml_args.Scheduler.warmup_ratio)

    def WarmupExponentialLR(cur_iter):
        gamma = math.exp(math.log(lr_min / lr_max) / (T_max - warm_up_iter))
        if cur_iter < warm_up_iter:
            return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
        else:
            return lr_max * gamma ** (cur_iter - warm_up_iter)

    scheduler = LambdaLR(optimizer, lr_lambda=WarmupExponentialLR, last_epoch=last_step if yaml_args.resume else -1)

    if command_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[command_args.local_rank])

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean").cuda()

    if command_args.local_rank == 0:
        df_lr = pd.DataFrame(
            [WarmupExponentialLR(i) for i in range(T_max)],
            columns=["Learning Rate"],
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_lr, linewidth=2)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        lr_img_path = os.path.join(yaml_args.Save.img_save_dir, "lr_schedule.png")
        plt.savefig(lr_img_path, dpi=300)
        logger.info(f"Save learning rate schedule to {lr_img_path}")

    gc.collect()
    # torch.cuda.empty_cache()

    do_train(yaml_args, command_args, model, train_loader, val_loader, optimizer, scheduler, criterion)

    wandb.finish()


def get_args_from_yaml(yaml_path):
    conf = OmegaConf.load(yaml_path)

    return conf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="xx")
    command_args = parser.parse_args()

    yaml_args = get_args_from_yaml("para.yaml")

    main(yaml_args, command_args)
