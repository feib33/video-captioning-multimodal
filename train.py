# Official packages
import os
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, logging
import torch.multiprocessing as mp
from timeit import default_timer as timer
from torch.nn.parallel import DistributedDataParallel as DDP


# My packages
from model.mmt import MMT

from config.config import get_cfg_defaults
from utils.training_utils import *
from utils.ddp_utils import *


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def set_random_seed(x: int, deterministic=False):
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    np.random.seed(x)
    random.seed(x)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, world_size):
    model.train()
    total_loss = 0

    if is_main_rank():
        dataloader = tqdm(dataloader)

    for data in dataloader:
        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = data

        # Put data on gpu
        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = (src.to(device), src_key_padding_mask.to(device),
                                            tgt_input.to(device), tgt_output.to(device),
                                            tgt_mask.to(device), tgt_key_padding_mask.to(device))

        with autocast():
            out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)
            # Compute the loss
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))

        # DDP
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size


        # Backward and optimizer
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # Gather data and report
        total_loss += loss.detach()

    torch.cuda.synchronize(device)
    return total_loss/len(dataloader)


def val_one_epoch(model, dataloader, criterion, device, world_size):
    model.eval()
    total_loss = 0

    if is_main_rank():
        dataloader = tqdm(dataloader)

    for data in dataloader:

        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = data

        # Put data on gpu
        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = (src.to(device), src_key_padding_mask.to(device),
                                            tgt_input.to(device), tgt_output.to(device),
                                            tgt_mask.to(device), tgt_key_padding_mask.to(device))
        with torch.no_grad():
            out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)

            # Compute the loss
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))

            # DDP
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size

        # Gather data and report
        total_loss += loss.detach()

    return total_loss/len(dataloader)


def train(rank, cfg, _world_size):

    if torch.cuda.is_available() is False:
        raise EnvironmentError("No GPU found")

    # Set up the process groups
    setup(rank, _world_size)
    logging.set_verbosity_error()  # Get rid of annoying warning when downloading bert_tokenizer from huggingface
    device = torch.device("cuda")

    # Build up the model
    model = MMT(cfg, mode="train", device=rank).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")

    # Set up assistants
    """
    log =
    """
    early_stopping = EarlyStopping(verbose=True, path=os.path.join(cfg.CKP.PATH, cfg.CKP.NAME + f"early_stopping.pth"))
    writer = SummaryWriter(os.path.join(cfg.SYSTEM.LOG_DIR, cfg.CKP.NAME))

    # Build up dataset and dataloader
    if is_main_rank():
        print("==="*10)
        print("Loading dataset")
        print("==="*10)
    train_loader, val_loader, train_sampler = build_dataloader(tokenizer, cfg)

    # Wrap the model and set up utilities
    ddp_model = DDP(model, device_ids=[rank])
    lr = 1e-4
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, ddp_model.parameters()), lr=lr)
    criterion = SCELoss(0.5, 0.5, 0, tokenizer.vocab_size)
    scaler = GradScaler()

    # Start Training
    epoch = cfg.SYSTEM.NUM_EPOCH
    if is_main_rank():
        print("==="*10)
        print("Let's start training")
        print("==="*10)
    for cur_epoch in range(1, epoch + 1):
        train_sampler.set_epoch(cur_epoch)

        # Train one epoch
        if is_main_rank():
            print(f" ++++ Epoch {cur_epoch}: ++++")
            start_time = timer()
        train_loss = train_one_epoch(ddp_model, train_loader, optimizer, scaler, criterion, device, _world_size)
        if is_main_rank():
            end_time = timer()
        # Validate one epoch
        val_loss = val_one_epoch(ddp_model, val_loader, criterion, device, _world_size)

        if is_main_rank():
            print(f"Train loss: {train_loss}, Val loss: {val_loss}, Spent {(end_time - start_time):.3f}s ")
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={'train loss': train_loss,
                                                'val loss': val_loss},
                               global_step=cur_epoch)
            print("="*5 + "Go to next epoch" + "="*5)

        # Save checkpoint
        ckp_name = os.path.join(cfg.CKP.PATH, cfg.CKP.NAME + f"_epoch{cur_epoch}.pth")
        if is_main_rank():
            if cur_epoch % 5 == 0:
                print("Saving checkpoint ...")
                torch.save(ddp_model.module.state_dict(), ckp_name)
            if early_stopping.early_stop:
                print(f" ### Early stopping at {cur_epoch} ###")
                break
    cleanup()
    print("Training completed! Please check the tensorbard")


if __name__ == "__main__":
    # get config
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config/config.yaml")
    cfg.freeze()

    # Preparation
    set_random_seed(114514)
    use_ddp = cfg.SYSTEM.DDP
    if use_ddp:
        world_size = cfg.SYSTEM.NUM_GPUS
        mp.spawn(train,
                 args=(cfg, world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        train(0, cfg, 1)  # only rank 0 is used for single gpu training




