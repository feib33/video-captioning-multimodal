import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader import Msrvtt, collate_fn





def build_dataloader(tokenizer, cfg):
    batch_size = cfg.SYSTEM.BATCH_SIZE
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataset = Msrvtt(feat_path=cfg.DATASET.FEAT.MT_PATH,
                           cap_path=cfg.DATASET.CAPTION.PATH_TO_TRAIN,
                           split="train", tokenizer=tokenizer)
    val_dataset = Msrvtt(feat_path=cfg.DATASET.FEAT.MT_PATH,
                         cap_path=cfg.DATASET.CAPTION.PATH_TO_VALIDATE,
                         split="validate", tokenizer=tokenizer)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              persistent_workers=True, num_workers=nw, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            persistent_workers=True, num_workers=nw, collate_fn=collate_fn)

    return train_loader, val_loader, train_sampler


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, ignore_index, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = -torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """This class is from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
