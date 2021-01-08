import torch.optim.lr_scheduler

class DummyScheduler:
    def __init__(self, lr):
        self.lr = lr
    def get_lr(self):
        return self.lr
    def step(self, epoch=None):
        pass

def scheduler_from_cfg(cfg, optimizer):
    if cfg.TRAINER.LR_SCHEDULE == 'constant':
        return DummyScheduler(cfg.TRAINER.LR)
    if cfg.TRAINER.LR_SCHEDULE == 'cosine_anneal':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg.TRAINER.SCHEDULER.T_0, cfg.TRAINER.SCHEDULER.T_MULT)
    if cfg.TRAINER.LR_SCHEDULE == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAINER.SCHEDULER.STEP_SIZE, cfg.TRAINER.SCHEDULER.GAMMA)
    else:
        raise Exception(f'Unknown scheduler {cfg.TRAINER.LR_SCHEDULE}')