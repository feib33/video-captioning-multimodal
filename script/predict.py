import torch
from model.mmt import MMT
from config.config import get_cfg_defaults
from torch.utils.data import DataLoader
from dataloader import Msrvtt, collate_fn
from model.cap_generator import eval_dataset


cfg = get_cfg_defaults()
cfg.merge_from_file("./config/config.yaml")
cfg.freeze()
device = "cuda"

model = MMT(cfg, "test", device)
model.load_state_dict(torch.load(cfg.PREDICT.CHECKPOINT))
test_dataset = Msrvtt(cfg.DATASET.FEAT.MT_PATH, cfg.DATASET.CAPTION.PATH_TO_TEST, "test", model.tokenizer, True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=collate_fn)

eval_dataset(model, test_dataset, test_loader, 30, device, model.tokenizer)
