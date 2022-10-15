"""
    encoder_input: (30, 2048)    [B, S, feat_size]
    decoder_input: ["[SOE]", "hello", "world"]
    decoder_output: ["hello", "world", "[EOE]"]

"""
import sys

import numpy
import numpy as np
import torch.nn as nn
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
    if is_main_rank():
        print(f"{len(vid2path)} video found for {split}")
    return vid2path


def create_vid2cap(cap_path, split):
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
    if is_main_rank():
        print(f"{len(vid2cap)} data will be used for {split}")
    return vid2cap


def collate_fn(batch):
    """
    Padding the src and tgt, then collate them into batch tensors with padding mask and tgt_mask
    :return: src, src_key_padding_mask, tgt, tgt_mask, tgt_key_padding_mask
    """

    features, captions = ([feat for feat, _ in batch], [cap for _, cap in batch])
    assert len(features) == len(captions), "The number of features is not equivalent to the number of captions "

    # padding to src and set up src and src_key_padding_mask (str -> tensor)
    feat_np_list = [np.load(path) for path in features]  # len(feat_np_list) = batch_size
    max_len = max([_feat.shape[0] for _feat in feat_np_list])
    batch_size = len(batch)
    feat_ts = torch.zeros((batch_size, max_len, 768)) # TBD change 2048 to cfg.MODEL.FEAT_SIZE
    feat_key_padding_mask = torch.ones((batch_size, max_len))
    for i, feat_np in enumerate(feat_np_list):
        feat_len = feat_np.shape[0]
        feat_ts[i, :feat_len, :] = torch.from_numpy(feat_np)
        feat_key_padding_mask[i, feat_len:] = 0
    feat_key_padding_mask = create_key_padding_mask(feat_key_padding_mask)

    # padding to tgt and set up tgt, tgt_mask, tgt_key_padding_mask (str -> tensor)
    cap_in = [cap[:-1] for cap in captions]  # Get rid of [EOE]
    max_len = max([len(cap) for cap in cap_in])
    for cap in cap_in:
        if len(cap) < max_len:
            diff = max_len - len(cap)
            cap.extend([0] * diff)
    cap_in = torch.tensor(cap_in)
    cap_mask = create_mask(cap_in)  # (T, T) since mask will be broadcasted to each sentence in the batch then it doesn't need to be (N*num_heads, T, T)
    cap_key_padding_mask = create_key_padding_mask(cap_in)

    cap_out = [cap[1:] for cap in captions] # Get rid of [SOE]
    for cap in cap_out:
        if len(cap) < max_len:
            diff = max_len - len(cap)
            cap.extend([0] * diff)
    cap_out = torch.tensor(cap_out)
    return feat_ts, feat_key_padding_mask, cap_in, cap_out, cap_mask, cap_key_padding_mask


class Msrvtt(data.Dataset):
    def __init__(self, feat_path, cap_path, split, tokenizer):
        super(Msrvtt, self).__init__()
        assert split in ["train", "validate", "test"]
        self.split = split
        self.vid2path = create_vid2path(feat_path, split)
        self.vid2cap = create_vid2cap(cap_path, split)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.vid2cap)

    def __getitem__(self, idx): # TODO the output may be a tensor ? need to be checked
        vid, cap = self.vid2cap[idx]
        cap_ids_list = self.tokenizer.encode(cap)
        return self.vid2path[vid], cap_ids_list

"""
tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")

val_dataset = Msrvtt(feat_path='./data/MSR-VTT/feats/timesformer/howto100m',
                          cap_path="./data/MSR-VTT/annotations/train_val_videodatainfo.json",
                          split="validate", tokenizer=tokenizer)

val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False,
                        num_workers=0, collate_fn=lambda batch: collate_fn(batch, tokenizer))

for batch_idx, data in enumerate(val_loader):
    _, _, tgt_in, tgt_out, tgt_mask, tgt_key_padding_mask = data
    print(tgt_in)
    print(tgt_out)
    print(tgt_mask)
    print(tgt_key_padding_mask)
    break
"""

