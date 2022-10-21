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

# ===================================
# Config Model
# ===================================
_C.MODEL = CN()
_C.MODEL.NAME = "Multi-Modal Transformer"
_C.MODEL.TOKENIZER = "bert-base-uncased"
_C.MODEL.D_MODEL = 768
_C.MODEL.FEAT_SIZE = 2048
_C.MODEL.DROPOUT = 0.5
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
# Config checkpoint's name
# ------------------------------------
_C.CKP = CN()
_C.CKP.PATH = "./checkpoint"
_C.CKP.MODEL_NAME = _C.DATASET.FEAT.MOTION_FEAT + '_'
_C.CKP.NAME = _C.CKP.MODEL_NAME + f"bsz{_C.SYSTEM.BATCH_SIZE}_lr{_C.SYSTEM.LR}_"\
              f"E(nl{_C.MODEL.ENCODER.NUM_LAYERS}_dp{_C.MODEL.ENCODER.DROPOUT})_" \
              f"D(nl{_C.MODEL.DECODER.NUM_LAYERS}_dp{_C.MODEL.DECODER.DROPOUT})"


def get_cfg_defaults():
    return _C.clone()