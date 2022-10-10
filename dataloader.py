"""
    encoder_input: (30, 2048)    [B, S, feat_size]
    decoder_input: ["[SOE]", "hello", "world"]
    decoder_output: ["hello", "world", "[EOE]"]

"""
""" MSR-VTT dataset """
import os
import torch
import glob
import json
from torch.utils import data


def create_vid2path(feat_path, split):
    """
    Create a dict containing video id and its path
    :param feat_path: the path of extracted features containing "train" "validate" and "test" directories
    :param split: "train", "validate" or "test"
    :return: dict({video_id : video_path})
    """
    vid2path = {}
    path_list = glob.glob(os.path.join(feat_path, split, "*.npy"))
    for path in path_list:
        vid = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        vid2path.update({vid:path})
    return vid2path


def create_vid2cap(cap_path, split):
    with open(cap_path, 'r') as j_file:
        annotations = json.load(j_file)
    raw_videos = annotations["videos"]
    raw_captions = annotations["sentences"]

    # Only the video correlated with inputted split will be concerned
    vid2split = [vid["video_id"] for vid in raw_videos if vid["split"] == split]

    vid2cap = []
    for raw_cap in raw_captions:
        if raw_cap["video_id"] in vid2split:
            vid2cap.append((raw_cap["video_id"], raw_cap["caption"]))
    return vid2cap


def my_collate_fn(self):
    pass


class Msrvtt(data.Dataset):
    def __init__(self, feat_path, cap_path, split):
        super(Msrvtt, self).__init__()
        self.split = split
        self.vid2path = create_vid2path(feat_path, split)
        self.vid2cap = create_vid2cap(cap_path, split)

    def __len__(self):
        return len(self.vid2cap)

    def __getitem__(self, idx): # TODO the output may be a tensor ? need to be checked
        vid, cap = self.vid2cap[idx]
        return self.vid2path[vid], cap




