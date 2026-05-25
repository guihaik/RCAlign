from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import numpy as np
import torch

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
rgb_img = cv2.imread('./pic/lidar/CAM_BACK_RIGHT.png').astype(np.float32)/255.
# rgb_img = cv2.imread('n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg').astype(np.float32)/255.
input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0) # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers)
cam = EigenCAM(model=model, target_layers=target_layers)
# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# targets = [ClassifierOutputTarget(281)]

targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
cv2.imwrite('vis.png', visualization)

# You can also get the model outputs without having to re-inference
# model_outputs = cam.outputs