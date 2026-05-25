from nuscenes.eval.detection.evaluate import NuScenesEval

import tqdm

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.splits import create_splits_scenes

import os

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox

rain_val = \
    ['scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634',
     'scene-0635', 'scene-0636', 'scene-0637', 'scene-0638', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-1060', 'scene-1065', 'scene-1067']

sunny_val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0770', 'scene-0771',
     'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784',
     'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1066', 'scene-1068', 'scene-1069', 'scene-1070',
     'scene-1071', 'scene-1072', 'scene-1073']

night_val = \
    ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066',
     'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

day_val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972']


class CusNuscenesEval(NuScenesEval):
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 split_name='rain'):
        # super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)
        """
                Initialize a DetectionEval object.
                :param nusc: A NuScenes object.
                :param config: A DetectionConfig object.
                :param result_path: Path of the nuScenes JSON result file.
                :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
                :param output_dir: Folder to save plots and results to.
                :param verbose: Whether to print to stdout.
                """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.split_name = split_name

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = self.load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose, split_name=split_name)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def load_gt(self, nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, split_name='rain') -> EvalBoxes:
        """
        Loads ground truth boxes from DB.
        :param nusc: A NuScenes instance.
        :param eval_split: The evaluation split for which we load GT boxes.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The GT boxes.
        """
        # Init.
        if box_cls == DetectionBox:
            attribute_map = {a['token']: a['name'] for a in nusc.attribute}

        if verbose:
            print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
        # Read out all sample_tokens in DB.
        sample_tokens_all = [s['token'] for s in nusc.sample]
        assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

        # Only keep samples from this split.
        cond_splits = {
            'rain': rain_val,
            'sunny': sunny_val,
            'night': night_val,
            'day': day_val,
        }

        # Only keep samples from this split.
        splits = create_splits_scenes()

        # Check compatibility of split with nusc_version.
        version = nusc.version
        if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
            assert version.endswith('trainval'), \
                'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        elif eval_split in {'mini_train', 'mini_val'}:
            assert version.endswith('mini'), \
                'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        elif eval_split == 'test':
            assert version.endswith('test'), \
                'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        else:
            raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                             .format(eval_split))

        if eval_split == 'test':
            # Check that you aren't trying to cheat :).
            assert len(nusc.sample_annotation) > 0, \
                'Error: You are trying to evaluate on the test set but you do not have the annotations!'

        sample_tokens = []
        for sample_token in sample_tokens_all:
            scene_token = nusc.get('sample', sample_token)['scene_token']
            scene_record = nusc.get('scene', scene_token)
            if scene_record['name'] in cond_splits[split_name] and scene_record['name'] in splits[eval_split]:
                sample_tokens.append(sample_token)

        all_annotations = EvalBoxes()

        # Load annotations and filter predictions and annotations.
        tracking_id_set = set()
        for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

            sample = nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']

            sample_boxes = []
            for sample_annotation_token in sample_annotation_tokens:

                sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
                if box_cls == DetectionBox:
                    # Get label name in detection task and filter unused labels.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])
                    if detection_name is None:
                        continue

                    # Get attribute_name.
                    attr_tokens = sample_annotation['attribute_tokens']
                    attr_count = len(attr_tokens)
                    if attr_count == 0:
                        attribute_name = ''
                    elif attr_count == 1:
                        attribute_name = attribute_map[attr_tokens[0]]
                    else:
                        raise Exception('Error: GT annotations must not have more than one attribute!')

                    sample_boxes.append(
                        box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation['translation'],
                            size=sample_annotation['size'],
                            rotation=sample_annotation['rotation'],
                            velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                            num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                            detection_name=detection_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                            attribute_name=attribute_name
                        )
                    )
                elif box_cls == TrackingBox:
                    # Use nuScenes token as tracking id.
                    tracking_id = sample_annotation['instance_token']
                    tracking_id_set.add(tracking_id)

                    # Get label name in detection task and filter unused labels.
                    # Import locally to avoid errors when motmetrics package is not installed.
                    from nuscenes.eval.tracking.utils import category_to_tracking_name
                    tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                    if tracking_name is None:
                        continue

                    sample_boxes.append(
                        box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation['translation'],
                            size=sample_annotation['size'],
                            rotation=sample_annotation['rotation'],
                            velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                            num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                            tracking_id=tracking_id,
                            tracking_name=tracking_name,
                            tracking_score=-1.0  # GT samples do not have a score.
                        )
                    )
                else:
                    raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

            all_annotations.add_boxes(sample_token, sample_boxes)

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

        return all_annotations
