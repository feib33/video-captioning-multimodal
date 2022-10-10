from yacs.config import CfgNode as CN

_C = CN()
# ===================================
# Config System Information
# ===================================
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.DEVICE_ID = 1
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
_C.MODEL.ENCODER.DROPOUT = 0.1
_C.MODEL.ENCODER.ACTIVATION = "gelu"
_C.MODEL.ENCODER.BATCH_FIRST = True
_C.MODEL.ENCODER.NUM_LAYERS = 4

# -----------------------------------
# Config Model Decoder Part
# -----------------------------------
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.D_MODEL = 768
_C.MODEL.DECODER.NHEAD = 8
_C.MODEL.DECODER.DROPOUT = 0.1
_C.MODEL.DECODER.ACTIVATION = "gelu"
_C.MODEL.DECODER.BATCH_FIRST = True
_C.MODEL.DECODER.NUM_LAYERS = 4


def get_cfg_defaults():
    return _C.clone()