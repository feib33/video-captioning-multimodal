# Official packages
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from timeit import default_timer as timer
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.utils import logging as t_logging

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


def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, cfg):
    model.train()
    total_loss = 0

    dataloader = tqdm(dataloader)

    for batch_idx, data in enumerate(dataloader):
        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = data

        # Put data on gpu
        (src, src_key_padding_mask,
         tgt_input, tgt_output,
         tgt_mask, tgt_key_padding_mask) = (src.to(device), src_key_padding_mask.to(device),
                                            tgt_input.to(device), tgt_output.to(device),
                                            tgt_mask.to(device), tgt_key_padding_mask.to(device))

        if cfg.SYSTEM.AMP:
            with autocast():
                out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)
                # Compute the loss
                loss = criterion(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))

            # Backward and optimizer
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)
            # Compute the loss
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))

            """
            # DDP
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size
            """

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Gather data and report
        total_loss += loss.item()
    return total_loss/len(dataloader)


@torch.no_grad()
def val_one_epoch(model, dataloader, criterion, device, cfg):
    model.eval()
    total_loss = 0
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

        out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)
        loss = criterion(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))

        """
        # DDP
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
        """
        # Gather data and report
        total_loss += loss.item()

    return total_loss/len(dataloader)


def train(cfg):
    # ===========================================
    # Initialize utilities
    # ===========================================
    t_logging.set_verbosity_error()  # Get rid of annoying warning when downloading bert_tokenizer from huggingface
    device = torch.device("cuda")
    logger = build_logger(__name__)
    model_name = cfg.DATASET.FEAT.MOTION_FEAT + '_'
    ckp_name = model_name + f"bsz{cfg.SYSTEM.BATCH_SIZE}_lr{cfg.SYSTEM.LR}_" \
                            f"E(nl{cfg.MODEL.ENCODER.NUM_LAYERS}_dp{cfg.MODEL.ENCODER.DROPOUT})_" \
                            f"D(nl{cfg.MODEL.DECODER.NUM_LAYERS}_dp{cfg.MODEL.DECODER.DROPOUT})"
    early_stopping = EarlyStopping(verbose=True, path=os.path.join(cfg.CKP.PATH, ckp_name + f"_early_stopping_at_epoch"))
    writer = SummaryWriter(os.path.join(cfg.SYSTEM.LOG_DIR, ckp_name))
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")


    # ===========================================
    # Initialize model and loader
    # ===========================================
    model = MMT(cfg, mode="train", device=device).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    logger.info(f"===" * 10)
    logger.info(f"Loading dataset")
    logger.info(f"===" * 10)
    train_loader, val_loader = build_dataloader(tokenizer, cfg)

    # ===========================================
    # Initial hyper-parameters
    # ===========================================
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=cfg.SYSTEM.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    criterion = SCELoss(0.5, 0.5, 0, tokenizer.vocab_size)
    scaler = GradScaler()  # when use amp

    # ===========================================
    # Start to train
    # ===========================================
    epoch = cfg.SYSTEM.NUM_EPOCH
    logger.info("===" * 10)
    logger.info("Let's start training")
    logger.info("===" * 10)

    for cur_epoch in range(1, epoch + 1):

        # Train one epoch
        logger.info(f" ++++ Epoch {cur_epoch}: ++++")
        start_time = timer()
        if cfg.SYSTEM.AMP:
            train_loss= train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, cfg)
        else:
            train_loss= train_one_epoch(model, train_loader, optimizer, None, criterion, device, cfg)
        end_time = timer()

        # Validate one epoch
        val_loss= val_one_epoch(model, val_loader, criterion, device, cfg)
        scheduler.step()
        # Write down the training loss and val loss into tensorboard
        logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}, Spent {(end_time - start_time):.3f}s ")
        writer.add_scalars(main_tag='Loss/epoch',
                           tag_scalar_dict={'train loss': train_loss,
                                            'val loss': val_loss},
                           global_step=cur_epoch)
        writer.add_scalar('lr/epoch', optimizer.state_dict()['param_groups'][0]['lr'], cur_epoch)
        writer.add_scalar('Loss/lr', train_loss, optimizer.state_dict()['param_groups'][0]['lr'])

        # Save checkpoint
        early_stopping(val_loss, model, cur_epoch)
        ckp_name = os.path.join(cfg.CKP.PATH, ckp_name + f"_epoch{cur_epoch}.pth")
        if cur_epoch % 5 == 0:
            logger.info("Saving checkpoint ...")
            torch.save(model.state_dict(), ckp_name)
        if early_stopping.early_stop:
            logger.info(f" ### Early stopped ###")
            break
    logger.info("Training completed! Please check the TensorBoard")


def dist_train(rank, cfg, _world_size):

    if not torch.cuda.is_available():
        raise EnvironmentError("No GPU found")

    # Set up the process groups
    setup(rank, _world_size)
    t_logging.set_verbosity_error()  # Get rid of annoying warning when downloading bert_tokenizer from huggingface
    device = torch.device("cuda")
    model_name = cfg.DATASET.FEAT.MOTION_FEAT + '_'
    ckp_name = model_name + f"bsz{cfg.SYSTEM.BATCH_SIZE}_lr{cfg.SYSTEM.LR}_" \
                            f"E(nl{cfg.MODEL.ENCODER.NUM_LAYERS}_dp{cfg.MODEL.ENCODER.DROPOUT})_" \
                            f"D(nl{cfg.MODEL.DECODER.NUM_LAYERS}_dp{cfg.MODEL.DECODER.DROPOUT})"

    print("This is rank: ", rank)
    # logging
    if is_main_rank():
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        log_formatter = logging.Formatter(fmt='%(message)s')

        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(log_formatter)
        logger.addHandler(s_handler)

        f_handler = logging.FileHandler(filename='./log/output.log')
        f_handler.setFormatter(log_formatter)
        logger.addHandler(f_handler)

    # Build up the model
    model = MMT(cfg, mode="train", device=rank).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")

    # Set up assistants
    early_stopping = EarlyStopping(verbose=True, path=os.path.join(cfg.CKP.PATH, ckp_name + f"early_stopping.pth"))
    writer = SummaryWriter(os.path.join(cfg.SYSTEM.LOG_DIR, ckp_name))

    # Build up dataset and dataloader
    if is_main_rank():
        logger.info("==="*10)
        logger.info("Loading dataset")
        logger.info("==="*10)
    train_loader, val_loader, train_sampler = build_dataloader(tokenizer, cfg)

    # Wrap the model and set up utilities
    ddp_model = DDP(model, device_ids=[rank])
    lr = 1e-4
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, ddp_model.parameters()), lr=lr)
    criterion = SCELoss(0.5, 0.5, 0, tokenizer.vocab_size)
    if cfg.SYSTEM.AMP:
        scaler = GradScaler()

    # Start Training
    epoch = cfg.SYSTEM.NUM_EPOCH
    if is_main_rank():
        logger.info("===" * 10)
        logger.info("Let's start training")
        logger.info("===" * 10)


    for cur_epoch in range(1, epoch + 1):
        train_sampler.set_epoch(cur_epoch)

        # Train one epoch
        if is_main_rank():
            logger.info(f" ++++ Epoch {cur_epoch}: ++++")
            start_time = timer()

        if cfg.SYSTEM.AMP:
            train_loss, _ = train_one_epoch(ddp_model, train_loader, optimizer, scaler, criterion, device, _world_size)
        else:
            train_loss, _ = train_one_epoch(ddp_model, train_loader, optimizer, None, criterion, device, _world_size)

        if is_main_rank():
            end_time = timer()
        # Validate one epoch
        val_loss, _ = val_one_epoch(ddp_model, val_loader, criterion, device, _world_size)

        # Write down the training loss and val loss into tensorboard
        if is_main_rank():
            logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}, Spent {(end_time - start_time):.3f}s ")
                        #f"Loss list: {loss_list} , Val loss list: {vloss_list}")
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={'train loss': train_loss,
                                                'val loss': val_loss},
                               global_step=cur_epoch)

        # Save checkpoint
        early_stopping(val_loss, ddp_model.module)
        ckp_name = os.path.join(cfg.CKP.PATH, ckp_name + f"_epoch{cur_epoch}.pth")
        if is_main_rank():
            if cur_epoch % 5 == 0:
                logger.info("Saving checkpoint ...")
                torch.save(ddp_model.module.state_dict(), ckp_name)
            if early_stopping.early_stop:
                logger.info(f" ### Early stopping at {cur_epoch} ###")
                break
    cleanup()
    logger.info("Training completed! Please check the tensorbard")


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
        mp.spawn(dist_train,
                 args=(cfg, world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        train(cfg)  # only rank 0 is used for single gpu training




