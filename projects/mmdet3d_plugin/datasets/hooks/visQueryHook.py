# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from projects.mmdet3d_plugin.datasets.hooks.utils import is_parallel

__all__ = ['VisQueryHook']


@HOOKS.register_module()
class VisQueryHook(Hook):
    """ """

    def __init__(self):
        super().__init__()

    def before_train_iter(self, runner):
        print(runner.model)

    # def after_train_iter(self, runner):
    #     if is_parallel(runner.model.module):
    #         # print(runner.model)
    #         model_sample = runner.model.module
    #     else:
    #         # print(runner.model)
    #         model_sample = runner.model
    #
    #     for name, param in model_sample.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     import pdb; pdb.set_trace()