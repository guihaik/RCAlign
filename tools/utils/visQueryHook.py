import os
import torch
import matplotlib.pyplot as plt
class visQueryHook:
    def __init__(self, model):
        self.model = model

        # DeformableMulFeatureAggregationCuda
        # fusion_query, torch.stack([img_output, radar_output])
        #
        self.path = './pic/query_vis'
        self.query = model.module.pts_bbox_head.transformer.decoder.layers[-1].attentions[1]
        self.idx = 0

    def regist_forward_hooks(self):
        self.query.register_forward_hook(self.plot_cross_attention)

    def plot_cross_attention(self, module, input, output):
        _, img_radar_output = output
        img_output = img_radar_output[0].squeeze()
        radar_output = img_radar_output[1].squeeze()

        s = torch.softmax(torch.matmul(img_output, radar_output.T),dim=-1)
        s = s.detach().cpu().numpy()
        fig, ax = plt.subplots()
        cax = ax.imshow(s, cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label('Weight Values')
        plt.savefig(os.path.join(self.path, str(self.idx)+'.png'))
        self.idx+=1
        # plt.show()