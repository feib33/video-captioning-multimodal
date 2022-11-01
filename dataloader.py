"""
    encoder_input: (30, 2048)    [B, S, feat_size]
    decoder_input: ["[SOE]", "hello", "world"]
    decoder_output: ["hello", "world", "[EOE]"]

"""
import sys

import numpy
import numpy as np
import torch.nn as nn
import logging
""" MSR-VTT dataset """
import os
import torch
import glob
import json


from torch.utils import data
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.ddp_utils import is_main_rank
from utils.data_utils import tokenize, create_mask, create_key_padding_mask


def add_pad(x: list, max_len: int):
    if len(x) < max_len:
        return x.append(0)


def create_vid2path(feat_path, split):
    """
    Create a dict containing video id and its path
    :param feat_path: the path of extracted features containing "train" "validate" and "test" directories
    :param split: "train", "validate" or "test"
    :return: dict({video_id : video_path})
    """
    assert os.path.exists(feat_path), f"The feat path '{feat_path}' is not existed"
    vid2path = {}
    path_list = glob.glob(os.path.join(feat_path, split, "*.npy"))
    for path in path_list:
        vid = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        vid2path.update({vid: path})
    """
    if is_main_rank():
        logging.info(f"{len(vid2path)} video found for {split}")
    """
    return vid2path


def create_vid2cap(cap_path, split):
    """
    Create a dictionary including vid_id and its caption
    :param cap_path: path to the annotation json file
    :param split: "train", "validate" or "test"
    :return: [(video_id, cap)]
    """
    assert os.path.exists(cap_path), f"The cap path '{cap_path}' is not existed"
    with open(cap_path, 'r') as j_file:
        annotations = json.load(j_file)
    raw_videos = annotations["videos"]
    raw_captions = annotations["sentences"]
    # Only the video which is correlated with inputted split will be concerned
    vid2split = [vid["video_id"] for vid in raw_videos if vid["split"] == split]
    vid2cap = []
    for raw_cap in raw_captions:
        if raw_cap["video_id"] in vid2split:
            vid2cap.append((raw_cap["video_id"], raw_cap["caption"]))
    """
    if is_main_rank():
        logging.info(f"{len(vid2cap)} data will be used for {split}")
    """
    return vid2cap


def return_feats_and_masks(feat_np_list, bsz, max_len):
    """
    Convert feat numpy to tensor and pad itself
    :param feat_np_list:  inputted feature numpy list
    :param bsz: batch size
    :param max_len: max length of feat along the dim 0 in the batch
    :param ids: video id list in the batch
    :param eval_by_metric: evaluate whole dataset by metric or not
    :return: src(padded) : (B, max_len, feat_size), src_key_padding_mask
    """
    feat_ts = torch.zeros((bsz, max_len, 2048))  # TBD change 2048 to cfg.MODEL.FEAT_SIZE
    feat_key_padding_mask = torch.ones((bsz, max_len))
    for i, feat_np in enumerate(feat_np_list):
        feat_len = feat_np.shape[0]
        feat_ts[i, :feat_len, :] = torch.from_numpy(feat_np)
        feat_key_padding_mask[i, feat_len:] = 0
    feat_key_padding_mask = create_key_padding_mask(feat_key_padding_mask)

    """
    feat_ts = [None] * bsz
    feat_kpm = [None] * bsz

    for i, feat_np in enumerate(feat_np_list):
        # Initialization
        feat_temp = torch.zeros((max_len, 768))  # TBD change 2048 to cfg.MODEL.FEAT_SIZE
        feat_kpm_temp = torch.ones(max_len)
        feat_len = feat_np.shape[0]

        # Substitution
        feat_temp[:feat_len, :] = torch.from_numpy(feat_np)
        feat_kpm_temp[feat_len:] = 0

        # Appending
        feat_ts[i] = feat_temp
        feat_kpm[i] = feat_kpm_temp

    assert len(feat_ts) == len(feat_kpm) == len(ids), f"The number of feats is not identical with ids"
    assert None not in feat_ts and None not in feat_kpm, f"The src tensors are not padded successfully"
    feat_ts_dict = dict(zip(ids, feat_ts)) if eval_by_metric else None
    feat_kpm_dict = dict(zip(ids, feat_kpm)) if eval_by_metric else None
    feat_ts = torch.stack(feat_ts)
    feat_kpm = torch.stack(feat_kpm)
    feat_kpm = create_key_padding_mask(feat_kpm)
    """
    return feat_ts, feat_key_padding_mask


def collate_fn(batch):
    """
    Padding the src and tgt, then collate them into batch tensors with padding mask and tgt_mask
    :batch: [(path1, cap1, vid1), (path1, cap2, vid1), ...]
    :return: src, src_key_padding_mask, tgt, tgt_mask, tgt_key_padding_mask
    """
    is_prediction = True if len(batch[0]) == 2 else False
    vids = [vid for _, _, vid in batch] if not is_prediction else [vid for _, vid in batch]

    # padding to src and set up src and src_key_padding_mask (numpy -> tensor)
    feat_np_list = [np.load(path) for path, _, _ in batch] if not is_prediction else [np.load(path) for path, _ in batch]# len(feat_np_list) = batch_size
    max_len = max([_feat.shape[0] for _feat in feat_np_list])
    batch_size = len(batch)
    feat_ts, feat_key_padding_mask = return_feats_and_masks(feat_np_list, batch_size, max_len)
    if is_prediction:
        return feat_ts, feat_key_padding_mask, vids
    # padding to tgt and set up tgt, tgt_mask, tgt_key_padding_mask (str -> tensor)
    cap_in = [cap[:-1] for _, cap, _ in batch]  # Get rid of [EOE]
    max_len = max([len(cap) for cap in cap_in])
    for cap in cap_in:
        if len(cap) < max_len:
            diff = max_len - len(cap)
            cap.extend([0] * diff)
    cap_in = torch.tensor(cap_in)
    cap_mask = create_mask(cap_in)  # (T, T) since mask will be broadcasted to each sentence in the batch then it doesn't need to be (N*num_heads, T, T)
    cap_key_padding_mask = create_key_padding_mask(cap_in)

    cap_out = [cap[1:] for _, cap, _ in batch] # Get rid of [SOE]
    for cap in cap_out:
        if len(cap) < max_len:
            diff = max_len - len(cap)
            cap.extend([0] * diff)
    cap_out = torch.tensor(cap_out)
    return feat_ts, feat_key_padding_mask, cap_in, cap_out, cap_mask, cap_key_padding_mask, vids


class Msrvtt(data.Dataset):
    def __init__(self, feat_path, cap_path, split, tokenizer, is_prediction=False):
        super(Msrvtt, self).__init__()
        assert split in ["train", "validate", "test"]
        self.split = split
        self.is_prediction = is_prediction
        self.vid2path = create_vid2path(feat_path, split)
        self.vid2cap = create_vid2cap(cap_path, split)
        self.tokenizer = tokenizer

    def __len__(self):
        if self.is_prediction:
            return len(self.vid2path)  # len == 497 in val predicting (decode the src) else len == 9940
        else:
            return len(self.vid2cap)

    def __getitem__(self, idx): # TODO the output may be a tensor ? need to be checked
        if self.is_prediction:
            vid = list(self.vid2path.keys())[idx]
            return self.vid2path[vid], vid
        else:
            vid, cap = self.vid2cap[idx]
            cap_ids_list = self.tokenizer.encode(cap)
            return self.vid2path[vid], cap_ids_list, vid

    def create_vid_feat_dict(self):
        """
        Create a pair of video_id - feature numpy
        :return: {vid1: feat1, vid2: feat2, ...}
        """
        if self.split not in ["validate", "test"]:
            raise TypeError(f"The split of dataset must be 'validate' or 'test'")
        vid_feat_pair = {}
        for vid, path in self.vid2path.items():
            if vid not in vid_feat_pair.keys():
                vid_feat_pair.update({vid: np.load(path)})
            else:
                raise KeyError(f"More than 2 feature numpy exists for {vid}")
        return vid_feat_pair

    def create_vid_mul_captions_dict(self):
        """
        :return: {vid1: [cap1, cap2, ...], vid2: [cap1, cap2, ...], ...}
        """
        if self.split not in ["validate", "test"]:
            raise TypeError(f"The split of dataset must be 'validate' or 'test'")
        vid_mul_captions = {}
        for vid, cap in self.vid2cap:
            if vid not in vid_mul_captions.keys():
                vid_mul_captions.update({vid: [cap]})
            else:
                vid_mul_captions[vid].append(cap)
        return vid_mul_captions

"""
tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")

val_dataset = Msrvtt(feat_path='./data/MSR-VTT/feats/timesformer/howto100m',
                          cap_path="./data/MSR-VTT/annotations/train_val_videodatainfo.json",
                          split="validate", tokenizer=tokenizer)

val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

for batch_idx, data in enumerate(val_loader):
    src, src_kmp, tgt_in, tgt_out, tgt_mask, tgt_key_padding_mask, ids = data
    print(ids)
    print(src)
    print(src_kmp)
    print(tgt_in)
    print(tgt_out)
    print(tgt_mask)
    print(tgt_key_padding_mask)
    break
"""

