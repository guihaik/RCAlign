import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
out_dir = './pic/result_vis/'

result_json = "val/work_dirs/main/stream_petr_r50_radar_deform_704_bs4_seq_60e/Tue_May__7_16_24_42_2024/pts_bbox/results_nusc"
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[:2]):
    if use_gt:
        nusc.render_sample(token, out_path = "./pic/result_vis/"+token+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = "./pic//result_vis/"+token+"_pred.png", verbose=False)

