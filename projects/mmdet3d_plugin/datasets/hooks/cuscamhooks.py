import os

import torch
import sys
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class CamHook(Hook):
    def __init__(self, interval=50):
        self.interval = interval

    def after_val_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            print(runner.iter+1)

    def after_val_epoch(self, runner):
        print(runner)
