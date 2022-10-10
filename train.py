import torch
import numpy
from transformers import AutoTokenizer


from model.mmt import MMT
from utils.data_utils import tokenize
from config.config import get_cfg_defaults

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config/config.yaml")
    cfg.freeze()

    model = MMT(cfg, mode="train")

    src = numpy.load("./data/sample/resnet152_fps3/video0.npy")
    src = torch.tensor(src, dtype=torch.float)

    tgt = "a car is shown"

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert_tokenizer")

    # at here, since this sample is unbatched, this should be (T) instead of (1, T)
    tgt_dict = tokenize(tokenizer, tgt, max_length=8)

    tgt_id = tgt_dict["input_ids"][0]

    tgt_key_padding_mask = tgt_dict["attention_mask"][0]
    tgt_key_padding_mask = (tgt_key_padding_mask == 0)

    src = src.unsqueeze(0)  # Delete make a batch size
    tgt_id = tgt_id.unsqueeze(0)  # Delete make a batch size
    tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(0)  # Delete make a batch size

    logit = model(src, tgt_id, tgt_key_padding_mask)
    print("logit:", logit)

