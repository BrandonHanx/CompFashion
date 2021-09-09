from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()
_C.DATASETS.CROP = False
_C.DATASETS.SUB_CATS = None

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.IMG_MODEL = "resnet50"
_C.MODEL.TEXT_MODEL = "bigru"
_C.MODEL.COMP_MODEL = "tirg"
_C.MODEL.CORR_MODEL = "fd"
_C.MODEL.LOSS = "bbc"
_C.MODEL.BATCHMINER = "semihard"
_C.MODEL.VOCAB = "glove"
_C.MODEL.WEIGHT = "imagenet"
_C.MODEL.EXCEPT_KEYS = None
_C.MODEL.I_FREEZE = False
_C.MODEL.T_FREEZE = False


# -----------------------------------------------------------------------------
# GRU
# -----------------------------------------------------------------------------
_C.MODEL.GRU = CN()
_C.MODEL.GRU.EMBEDDING_SIZE = 512
_C.MODEL.GRU.NUM_UNITS = 512
_C.MODEL.GRU.VOCABULARY_SIZE = 12000
_C.MODEL.GRU.DROPOUT = 0.0
_C.MODEL.GRU.NUM_LAYER = 1

# -----------------------------------------------------------------------------
# BERT
# -----------------------------------------------------------------------------
_C.MODEL.BERT = CN()
_C.MODEL.BERT.POOL = True

# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1
_C.MODEL.RESNET.PRETRAINED = None

# -----------------------------------------------------------------------------
# Composition
# -----------------------------------------------------------------------------
_C.MODEL.COMP = CN()
_C.MODEL.COMP.METHOD = "base"
_C.MODEL.COMP.EMBED_DIM = 512

# -----------------------------------------------------------------------------
# Normalization Layer
# -----------------------------------------------------------------------------
_C.MODEL.NORM = CN()
_C.MODEL.NORM.SCALE = 4.0
_C.MODEL.NORM.LEARNABLE = True

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100

_C.SOLVER.LOG_PERIOD = 10
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.EVALUATE_PERIOD = 1
_C.SOLVER.GEN_EVALUATE_MODE = False

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.00004

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.SGD_MOMENTUM = 0.9

_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.GEN_TEST_MODE = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
