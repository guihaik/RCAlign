import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
import time
from mmdet3d.core.bbox import LiDARInstance3DBoxes
class Visualization:
    def __init__(self):
        self.point_clound_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    def plot_bev_point(self, points, norm=True, dir='feature_map/', prefix='pic'):
        savepath = os.path.join(prefix, dir)
        pc_range = self.point_clound_range
        points = points.detach().cpu()
        if norm:
            points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

        for i in range(len(points)):
            single_points = points[i]
            fig, ax = plt.subplots()
            ax.scatter(single_points[:, 0], single_points[:, 1])

            ax.set_xlim((pc_range[0], pc_range[3]))
            ax.set_ylim((-pc_range[1], -pc_range[4]))
            ax.set_aspect('equal', 'box')  # 使x轴和y轴的比例相等
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('BEV points')
            plt.savefig(f'{savepath}_bevpoint_{i}.png')
            plt.show()

    def get_bbox(self, bbox, dir='feature_map/', prefix='pic'):
        savepath = os.path.join(dir, prefix)
        if not isinstance(bbox[0], LiDARInstance3DBoxes):
            lidar_bbox3d = []
            for i in range(len(bbox)):
                bbox_i = LiDARInstance3DBoxes(bbox[i], box_dim=9, origin=(0.5, 0.5, 0.5))
                lidar_bbox3d.append(bbox_i)
            bbox = lidar_bbox3d
        for i in range(len(bbox)):
            bbox_bev = bbox[i].bev.detach().cpu().numpy()    #N*5
            # bbox_bev[:, :2] = bbox_bev[:, :2]-self.point_clound_range[:2]
            plane_x_range = (self.point_clound_range[0], self.point_clound_range[3])
            plane_y_range = (-self.point_clound_range[1], -self.point_clound_range[4])
            fig, ax = plt.subplots()

            for box in bbox_bev:
                x, y, l, w, angle = box
                corners = np.array([
                    [-w / 2, -l / 2],
                    [-w / 2, l / 2],
                    [w / 2, l / 2],
                    [w / 2, -l / 2],
                    [-w / 2, -l / 2]
                ])

                # 旋转角度
                rotation_matrix = np.array([
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
                ])

                rotated_corners = np.dot(corners, -rotation_matrix.T)

                # 平移中心点坐标
                rotated_corners += np.array([x, y])

                # 绘制框
                ax.plot(rotated_corners[:, 0], rotated_corners[:, 1])
            ax.set_xlim(plane_x_range)
            ax.set_ylim(plane_y_range)
            ax.set_aspect('equal', 'box')  # 使x轴和y轴的比例相等
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('BEV Boxes')
            plt.savefig(f'{savepath}_bevbbox_{i}.png')
            plt.show()


    def featuremap_2_heatmap(self, feature_map):
        assert isinstance(feature_map, torch.Tensor)
        feature_map = feature_map.detach()
        heatmap = feature_map[:,0,:,:]*0
        for c in range(feature_map.shape[1]):
            heatmap+=feature_map[:,c,:,:]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def draw_feature_map(self, features, dir='feature_map', name=None, prefix='./pic'):
        save_path = os.path.join(prefix, dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, name)
        if isinstance(features, torch.Tensor):
            i = 0
            for heat_maps in features:
                heat_maps=heat_maps.unsqueeze(0)
                heatmap = self.featuremap_2_heatmap(heat_maps)
                # 这里的h,w指的是你想要把特征图resize成多大的尺寸
                # heatmap = cv2.resize(heatmap, (h, w))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img)
                plt.savefig(f'{save_path}_{i}.png')
                plt.show()
                i+=1
        else:
            for featuremap in features:
                heatmaps = self.featuremap_2_heatmap(featuremap)
                # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                for heatmap in heatmaps:
                    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    # superimposed_img = heatmap * 0.5 + img*0.3
                    superimposed_img = heatmap
                    plt.imshow(superimposed_img)
                    plt.show()
                    # cv2.imshow("1",superimposed_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                    # i=i+1

    def showFeature2Img(self, imgs, feats):
        from PIL import Image
        if feats.dim()==5:
            B, N, H, W, C = feats.shape
            for i in range(N):
                feat = feats[:, i].squeeze(0)
                img = cv2.imread(imgs[i])

                feat = feat.detach().cpu()

                image = Image.fromarray(img.astype(np.uint8)).convert('RGB')

                height, width = image.size

                heatmap = torch.mean(feat, dim=0)
                heatmap = heatmap.numpy()
                heatmap /= np.max(heatmap)

                heatmap = cv2.resize(heatmap, (height, width))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)

                superimg = heatmap * 0.4 + np.array(image)[:, :, ::-1]
                # cv2.imshow(f'{i}', superimg)
                cv2.imwrite(f'{i}.png', superimg)

    def draw_graph(self, node, edges):
        if isinstance(edges, torch.Tensor):
            edges = edges.transpose(0, 1).tolist()
        save_path = '../result_vis/graph/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 创建一个空的无向图
        G = nx.Graph()

        # 添加点的数量
        # node=node
        G.add_nodes_from(range(node))

        # 添加所有的边
        # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
        G.add_edges_from(edges)

        # 绘制图形
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G, k=0.3, seed=42)  # 选择节点位置算法，这里使用了Spring layout
        nx.draw_networkx(G, pos, with_labels=True, node_size=25, node_color='lightblue', font_size=2, font_weight='bold',
                edge_color='gray', width=0.3)
        plt.title("Graph Visualization")
        plt.savefig(save_path + str(int(time.time())) + '.svg', dpi=300, format="svg")
        plt.show()
    
if __name__ == '__main__':
    vis = Visualization()
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    vis.draw_graph(6, edges)