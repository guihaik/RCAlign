import torch
from mmcv.runner.checkpoint import weights_to_cpu, get_state_dict


def conv_next():
    load_from = './ckpts/convnext_base_22k_224_mmdet.pth'

    model = torch.load(load_from, 'cpu')

    save_model = {}
    save_model['state_dict'] = {}

    for key, value in model['model'].items():
        new_key = 'img_backbone.' + key
        save_model['state_dict'][new_key] = value

    print(model)
    torch.save(save_model, './ckpts/convnext_base_22k_224_mmdet.pth')


def vit():
    import torch

    pretrain_dict = torch.load('./ckpts/eva02_L_coco_det_sys_o365.pth', map_location=torch.device('cpu'))
    pretrain_dict = pretrain_dict["model"]
    print(pretrain_dict.keys())
    remapped_dict = {}
    for k, v in pretrain_dict.items():
        if "backbone.net" in k:
            remapped_dict[k.replace("backbone.net.", "img_backbone.")] = v
        if "backbone.simfp" in k:
            remapped_dict[k.replace("backbone.", "img_backbone.adapter.")] = v
    torch.save(remapped_dict, './ckpts/eva02_L_coco_det_sys_o365_remapped.pth')


if __name__ == '__main__':
    vit()
