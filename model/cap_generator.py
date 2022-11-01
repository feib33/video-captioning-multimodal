import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from config.config import get_cfg_defaults
from utils.data_utils import *
from dataloader import Msrvtt, collate_fn
from torch.utils.data import DataLoader
from model import mmt
from metric.metric import eval
from dataloader import return_feats_and_masks
# from metric.metric import *



def eval_dataset(model, dataset, dataloader, sen_len, device, tokenizer):
    dataloader = tqdm(dataloader)
    vid_mul_caps = dataset.create_vid_mul_captions_dict()
    vid2sentence = dict(zip(list(vid_mul_caps.keys()), [None]*len(list(vid_mul_caps.keys()))))
    for batch_idx, data in enumerate(dataloader):
        src, src_key_padding_mask, vid = data
        batch_vid2ids = greedy_search(model, src.to(device), src_key_padding_mask.to(device), vid, sen_len, device)
        vids = batch_vid2ids.keys()
        sens = tokenizer.batch_decode(batch_vid2ids.values(), skip_special_tokens=True)
        batch_vid2sen = {k: [v] for k, v in zip(vids, sens)}
        vid2sentence.update(batch_vid2sen)

    """
    vid2sentences =  {vid1: [cap1], vid2: [cap2], vid3: [cap3], ...}
    vid_mul_caps = {vid1: [cap1, cap2, ... ,cap20],  }
    """
    assert len(vid2sentence) == len(vid_mul_caps), f"Make sure each video has a generated sentence and ground-truths"
    return eval(vid_mul_caps, vid2sentence)


def greedy_search(model, src, src_key_padding_mask, video_ids, sen_len, device):
    """
    Convert a batch of feature numpy to video ids using greedy search
    :param model: mmt
    :param src: feature tensor (B, S, 768)
    :param src_key_padding_mask: Boolean tensor (B, S)
    :param video_ids: a list of video ids (len = bsz)
    :param sen_len: max length of output sentence
    :param device: cpu or gpu
    :return: {vid1: cap1, vid2: cap2, ...}
    """
    softmax = torch.nn.Softmax(dim=-1)
    assert src.dim() == 3, f"feature dimension should be 3 (B, S, feat_size)"
    bsz = src.shape[0] if src.dim() == 3 else 1
    memory = model.encode(src, src_key_padding_mask)
    tgt = torch.ones(bsz, 1).fill_(101).type(torch.long).to(device)  # (B, 1) TBD change 101 to [CLS]

    eos_flag = [0] * bsz
    for _ in range(sen_len):
        tgt_mask = create_mask(tgt).to(device)
        out = model.decode(memory, tgt, tgt_mask, None)
        word_prob = softmax(out[:, -1])  # (B, vocab_size) this is one word's probability
        word = torch.argmax(word_prob, dim=-1)  # index of word in vocab (B)

        for i, w in enumerate(word):
            if eos_flag[i]:
                word[i] = 0  # TBD change 0 to [PAD]
            if w == 102 and not eos_flag[i]:  # TBD change 102 to [SEP]
                eos_flag[i] = 1

        tgt = torch.cat((tgt, word.unsqueeze(-1)), -1)

        if sum(eos_flag) >= bsz:
            break

    vid2ids = {}
    for i, vid in enumerate(video_ids):
        if vid not in vid2ids.keys():
            vid2ids.update({vid: tgt[i, :]})
        else:
            raise TypeError(f"{vid} should only have one caption")

    return vid2ids


"""
if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("../config/config.yaml")
    cfg.freeze()
    device = "cuda"
    model = mmt.MMT(cfg, "validate", device).to(device)
    model.load_state_dict(torch.load(cfg.PREDICT.CHECKPOINT))
    tokenizer = model.tokenizer
    val_dataset = Msrvtt(feat_path=cfg.DATASET.FEAT.MT_PATH,
                         cap_path=cfg.DATASET.CAPTION.PATH_TO_VALIDATE,
                         split="validate", tokenizer=tokenizer, is_prediction=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    eval_dataset(model, val_dataset, val_loader, 30, device, tokenizer)



for data in val_loader:
    (src, src_key_padding_mask,
     tgt_input, tgt_output,
     tgt_mask, tgt_key_padding_mask) = data

    (src, src_key_padding_mask,
     tgt_input, tgt_output,
     tgt_mask, tgt_key_padding_mask) = (src.to(device), src_key_padding_mask.to(device),
                                        tgt_input.to(device), tgt_output.to(device),
                                        tgt_mask.to(device), tgt_key_padding_mask.to(device))

    print(f'the tgt input shape is {tgt_input.shape}')
    print(tgt_input)
    out = model(src, src_key_padding_mask, tgt_input, tgt_mask, tgt_key_padding_mask)
    print(out.shape)
    T = out.shape[1]
    softmax = torch.nn.Softmax(dim=-1)
    sentence = []
    for i in range(T):
        word_ts = out[:, i, :]
        word_prob = softmax(word_ts)
        word = torch.argmax(word_prob, dim=-1)
        word = word.item()
        sentence.append(word)
    print(sentence)
    sentences = tokenizer.decode(sentence)
    print(sentences)
    break
"""
