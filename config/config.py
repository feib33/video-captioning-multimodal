from yacs.config import CfgNode as CN

_C = CN()
# ===================================
# Config System Information
# ===================================
_C.SYSTEM = CN()
_C.SYSTEM.DDP = False
_C.SYSTEM.AMP = False
_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.DEVICE_ID = 1
_C.SYSTEM.NUM_EPOCH = 30
_C.SYSTEM.BATCH_SIZE = 32
_C.SYSTEM.LR = 1e-4
_C.SYSTEM.OUTPUT_DIR = "./out"
_C.SYSTEM.LOG_DIR = "./log"
_C.SYSTEM.EVAL_BY_METRIC = True

# ===================================
# Config Model
# ===================================
_C.MODEL = CN()
_C.MODEL.NAME = "Multi-Modal Transformer"
_C.MODEL.TOKENIZER = "bert-base-uncased"
_C.MODEL.D_MODEL = 768
_C.MODEL.FEAT_SIZE = 2048
_C.MODEL.DROPOUT = 0.5
_C.MODEL.EMB_TRAIN = False
# -----------------------------------
# Config Model Encoder Part
# -----------------------------------
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.D_MODEL = 768
_C.MODEL.ENCODER.NHEAD = 8
_C.MODEL.ENCODER.DROPOUT = 0.3
_C.MODEL.ENCODER.ACTIVATION = "gelu"
_C.MODEL.ENCODER.BATCH_FIRST = True
_C.MODEL.ENCODER.NUM_LAYERS = 4

# -----------------------------------
# Config Model Decoder Part
# -----------------------------------
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.D_MODEL = 768
_C.MODEL.DECODER.NHEAD = 8
_C.MODEL.DECODER.DROPOUT = 0.3
_C.MODEL.DECODER.ACTIVATION = "gelu"
_C.MODEL.DECODER.BATCH_FIRST = True
_C.MODEL.DECODER.NUM_LAYERS = 4

# ------------------------------------
# Config dataset
# ------------------------------------
_C.DATASET = CN()
_C.DATASET.FEAT = CN()
_C.DATASET.FEAT.MOTION_FEAT = "timesformer"
_C.DATASET.FEAT.MT_PATH = "./data/MSR-VTT/feats/timesformer/howto100m"
_C.DATASET.CAPTION = CN()
_C.DATASET.CAPTION.PATH_TO_TRAIN = "./data/MSR-VTT/annotations/train_val_videodatainfo.json"
_C.DATASET.CAPTION.PATH_TO_VALIDATE = "./data/MSR-VTT/annotations/train_val_videodatainfo.json"
_C.DATASET.CAPTION.PATH_TO_TEST = "./data/MSR-VTT/annotations/test_videodatainfo.json"

# ------------------------------------
# Config inference
# ------------------------------------
_C.PREDICT = CN()
_C.PREDICT.CHECKPOINT = "/home/feib/video-captioning/backup/onegpu/cosineanealling_lr/ckp/timesformer_bsz32_lr0.0001_E(nl2_dp0.3)_D(nl4_dp0.3)_early_stopping.pth"


# ------------------------------------
# Config checkpoint's name
# ------------------------------------
_C.CKP = CN()
_C.CKP.PATH = "./checkpoint"
_C.CKP.MODEL_NAME = _C.DATASET.FEAT.MOTION_FEAT + '_'



def get_cfg_defaults():
    return _C.clone()