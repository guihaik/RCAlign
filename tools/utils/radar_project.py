import torch
import numpy as np
import mmcv
import cv2
import copy
from projects.mmdet3d_plugin.datasets.loading import LoadRadarPointsMultiSweeps, LoadImageFromFile
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from nuscenes.nuscenes import NuScenes

class RadarProject:
    def __init__(self, ann_file, first=False, radar_use_dims = [0, 1, 2, 5, 8, 9, 18]):
        # pass
        self.first = first
        self.ann_file = ann_file
        self.loadImage = LoadImageFromFile(to_float32=True)
        self.loadRadar = LoadRadarPointsMultiSweeps(load_dim=18, sweeps_num=5, use_dim=radar_use_dims, max_num=1200)
        self.loadlidar = LoadPointsFromFile( coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4])
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

        if self.first:
            self.data_infos = self.load_annotations(self.ann_file)
            self.data_infos['infos'] = self.data_infos['infos'][:100]
            self.data_infos['metadata'] = {'version': 'v1.0-trainval-sample100'}
            mmcv.dump(self.data_infos, './data/train_sample.pkl', file_format='pkl')
        else:
            self.data_infos = self.load_annotations(self.ann_file)
            self.data_infos = self.data_infos['infos']

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file, file_format='pkl')


    def invert_matrix_egopose_numpy(self, egopose):
        """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
        inverse_matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = egopose[:3, :3]
        translation = egopose[:3, 3]
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
        inverse_matrix[3, 3] = 1.0
        return inverse_matrix

    def convert_egopose_to_matrix_numpy(self, rotation, translation):
        transformation_matrix = np.zeros((4, 4), dtype=np.float32)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        transformation_matrix[3, 3] = 1.0
        return transformation_matrix

    def test_proj1(self, cam_info):
        cam2lidar_r = cam_info['sensor2lidar_rotation']
        cam2lidar_t = cam_info['sensor2lidar_translation']
        cam2lidar_rt = self.convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
        lidar2cam_rt = self.invert_matrix_egopose_numpy(cam2lidar_rt)

        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt)
        # intrinsics.append(viewpad)
        # extrinsics.append(lidar2cam_rt)
        # lidar2img_rts.append(lidar2img_rt)
        return lidar2img_rt, cam2lidar_rt

    def test_proj2(self, cam_info):
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info[
                          'sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
        return lidar2img_rt, camera2lidar

    def get_relate_data(self, data):
        cams = data['cams']
        for key, value in cams.items():
            img_path = value['data_path']
            img = cv2.imread(img_path)
            # cv2.imwrite(f'{key}.png', img)

        cams_front = cams['CAM_FRONT']
        lidar2img_rt, _ = self.test_proj1(cams_front)

        data['radar'] = data['radars']
        radar_data = self.loadRadar(data)['radar']
        return lidar2img_rt, radar_data, cams_front['data_path']

    def test_lidar2img(self):
        for data in self.data_infos:
            for cam_type, cam_info in data['cams'].items():
                lidar2img_rt1, cam2lidar_rt1 = self.test_proj1(cam_info)
                lidar2img_rt2, cam2lidar_rt2 = self.test_proj2(cam_info)
                print(lidar2img_rt1)
                print(cam2lidar_rt1)
                print(lidar2img_rt2)
                print(cam2lidar_rt2)

    def show_img_point(self, img_path, radar_point, name):
        img = cv2.imread(img_path)
        for i, p in enumerate(radar_point.T):
            img = cv2.circle(img, (int(p[0]),int(p[1])), 5, (255,0,0), -1)
        # cv2.imshow('img', img)
        cv2.imwrite(name, img)

    def getpoint2d(self, data, mat):
        radar_data = torch.cat([data, torch.ones_like(data[..., :1])], dim=-1).numpy()
        point2d = np.matmul(radar_data, mat.T)
        point2d = point2d.T
        depths = point2d[2, :]
        width = 1600
        height = 900
        point2d = point2d[:2, :] / np.clip(point2d[2:3, :], a_min=1e-5, a_max=10000000)
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, point2d[0, :] > 1)
        mask = np.logical_and(mask, point2d[0, :] < width - 1)
        mask = np.logical_and(mask, point2d[1, :] > 1)
        mask = np.logical_and(mask, point2d[1, :] < height - 1)
        point2d = point2d[:, mask]
        # point2d[2, :] = depths[mask]
        # point2d = point2d[:,:2]
        # print(point2d)
        return point2d

    def radar2img(self):
        for data in self.data_infos:
            data['pts_filename'] = data['lidar_path']
            token = data['token']
            lidar_data = self.loadlidar(data)['points']
            lidar_data = lidar_data.coord
            lidar2img_rt, radar_data, img_path = self.get_relate_data(data)
            radar_data = radar_data.tensor[:, :3]
            radar_point2d = self.getpoint2d(radar_data, lidar2img_rt)
            lidar_point2d = self.getpoint2d(lidar_data, lidar2img_rt)
            self.show_img_point(img_path, radar_point2d, 'radar.png')
            self.show_img_point(img_path, lidar_point2d, 'lidar.png')
            self.nus_proj(token, camera_channel='CAM_FRONT')

    def nus_proj(self, token, camera_channel):
        self.nusc.render_pointcloud_in_image(token, pointsensor_channel='LIDAR_TOP', camera_channel=camera_channel, render_intensity=True)
        self.nusc.render_pointcloud_in_image(token, pointsensor_channel='RADAR_FRONT', camera_channel = camera_channel)

    def nus_proj_bev(self, data):
        self.nusc.render_sample_data(data['radars']['RADAR_FRONT'][0]['sample_data_token'], nsweeps=5, underlay_map=True)

    def pro_img(self, results, diff=''):
        imgs = results['img']
        lidar2imgs = results['lidar2img']
        radar_data = results['radar'].coord
        token = results['sample_idx']
        cams_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for i, img in enumerate(imgs):
            img = np.ascontiguousarray(img)
            lidar2img = lidar2imgs[i]
            pts_4d = torch.cat([radar_data, radar_data.new_ones(radar_data[..., :1].size())], dim=-1)
            pts_2d = pts_4d @ lidar2img.T
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]

            pts_2d = pts_2d.T
            depths = pts_2d[2, :]
            width = img.shape[1]
            height = img.shape[0]
            # pts_2d = pts_2d[:2, :] / np.clip(pts_2d[2:3, :], a_min=1e-5, a_max=10000000)
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 0)
            mask = np.logical_and(mask, pts_2d[0, :] > 1)
            mask = np.logical_and(mask, pts_2d[0, :] < width - 1)
            mask = np.logical_and(mask, pts_2d[1, :] > 1)
            mask = np.logical_and(mask, pts_2d[1, :] < height - 1)
            pts_2d = pts_2d[:, mask]
            for j, p in enumerate(pts_2d.T):
                img = cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
            cv2.imwrite(f'img{i}{diff}.png', img)
            # self.nus_proj(token,cams_list[i])

    def draw_lidar_bbox3d_on_img(self, bboxes3d,
                                 raw_img,
                                 lidar2img_rt,
                                 img_metas,
                                 color=(0, 255, 0),
                                 thickness=1):
        img = raw_img.copy()
        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
             np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

        return self.plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

    def plot_rect3d_on_img(self, img,
                           num_rects,
                           rect_corners,
                           color=(0, 255, 0),
                           thickness=1):
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int32)
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), color, thickness,
                         cv2.LINE_AA)
        return img.astype(np.uint8)

if __name__=='__main__':
    debug_file = False      #sample file for debug
    if debug_file:
        data_root = '/opt/data/private/klh/Object_Detection/StreamPETR/data/nuscenes/nuscenes2d_temporal_infos_train.pkl'
        radar_proj = RadarProject(data_root, True)
    else:
        data_root = './data/train_sample.pkl'
        radar_proj = RadarProject(data_root)
    # radar_proj.test_lidar2img()
    radar_proj.radar2img()

    # results = 'dataloder_result'
    # radar_proj.pro_img(results, diff='str')

