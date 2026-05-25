import json

from nuscenes.utils import splits
from nuscenes import NuScenes
import os
import pickle
from tqdm import tqdm

def get_split(root_dir, version, split_name=None, return_token=False):
    val = splits.val
    des_path = os.path.join(root_dir, version, 'scene.json')

    with open (des_path, 'r') as file:
        scenes = json.load(file)

    count_scenes = {'rain':0, 'night':0, 'day':0, 'sunny':0}
    split_scenes = {'rain':[], 'night':[], 'day':[], 'sunny':[]}
    split_scenes_token = {'rain':[], 'night':[], 'day':[], 'sunny':[]}
    count_samples = {'rain':0, 'night':0, 'day':0, 'sunny':0}
    description_scenes = {'rain':[], 'night':[], 'day':[], 'sunny':[]}

    for scene in scenes:
        if scene['name'] in val:
            if 'rain' in scene['description'].lower():
                count_scenes['rain'] += 1
                split_scenes['rain'].append(scene['name'])
                split_scenes_token['rain'].append(scene['token'])
                description_scenes['rain'].append(scene['description'])
                count_samples['rain'] += scene['nbr_samples']
            else:
                count_scenes['sunny'] += 1
                split_scenes['sunny'].append(scene['name'])
                split_scenes_token['sunny'].append(scene['token'])
                description_scenes['sunny'].append(scene['description'])
                count_samples['sunny'] += scene['nbr_samples']

            if 'night' in scene['description'].lower():
                count_scenes['night'] += 1
                split_scenes['night'].append(scene['name'])
                split_scenes_token['night'].append(scene['token'])
                description_scenes['night'].append(scene['description'])
                count_samples['night'] += scene['nbr_samples']
            else:
                count_scenes['day'] += 1
                split_scenes['day'].append(scene['name'])
                split_scenes_token['day'].append(scene['token'])
                description_scenes['day'].append(scene['description'])
                count_samples['day'] += scene['nbr_samples']
    # print(count_scenes)
    # print(split_scenes)
    print(count_samples)

    print(split_name, split_scenes[split_name])

    if return_token:
        return split_scenes_token if split_name is None else split_scenes_token[split_name]
    else:
        return split_scenes if split_name is None else split_scenes[split_name]

def create_split_dataset(nusc_root, extra_tag, nuscenes_version, split_scenes, split_name):
    # 只对不同天气进行分割，不会对train和val进行分割
    dataroot = nusc_root
    set = 'val'
    dataset = pickle.load(open(dataroot + '/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
    dataset['metadata']['split_name'] = split_name
    split_set = {
        'infos': [],
        'metadata': dataset['metadata']
    }
    for id in tqdm(range(len(dataset['infos']))):  # sample iter

        info = dataset['infos'][id]

        if info['scene_token'] in split_scenes:
            split_set['infos'].append(info)

    extra_tag = f'{extra_tag}_{split_name}'
    print(f"Save {split_name} splits to {dataroot} {extra_tag}_infos_{set}.pkl, length of samples: {len(split_set['infos'])}")
    with open(dataroot + '/%s_infos_%s.pkl' % (extra_tag, set), 'wb') as fid:
        pickle.dump(split_set, fid)


if __name__ == '__main__':
    version = 'v1.0'
    version = f'{version}-trainval'
    # version = f'{version}-mini'
    # 若要使用mini，将trainval改为mini即可，但需要注意会覆盖掉已生成的trainval文件
    root_path = '/opt/data/private/klh/Object_Detection/StreamPETR_o_r_cha/data/nuscenes/'
    extra_tag = 'nuscenes2d_temporal'
    split_names = ['rain', 'sunny', 'night', 'day']
    for split_name in split_names:
        split_scenes = get_split(root_path, version, split_name, return_token=True)
        # create_split_dataset(root_path, extra_tag, version, split_scenes, split_name)
