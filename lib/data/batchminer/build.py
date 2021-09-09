from .semihard import SemiHardBatchMiner


def build_batchminer(cfg):
    if cfg.MODEL.BATCHMINER == "semihard":
        return SemiHardBatchMiner()
    return NotImplementedError
