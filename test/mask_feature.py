import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import cv2
from PIL import Image
from copy import deepcopy
import warnings
warnings.filterwarnings(action="ignore")

import sys
from pathlib import Path
sys.path.append(str(list(Path(__file__).parents)[1]))
from segment_anything import sam_model_registry, SamPredictor

# set image and mask path 
image_index = "2669"
image_path = f"/home/injo/solutions/mars/data/pascalvoc_2012/VOCdevkit/VOC2012/JPEGImages/2007_{image_index.zfill(6)}.jpg"
mask_path = f"/home/injo/solutions/mars/data/pascalvoc_2012/VOCdevkit/VOC2012/SegmentationObject/2007_{image_index.zfill(6)}.png"

# set sam's parameters
sam_checkpoint = "/home/injo/solutions/mars/data/sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:1" 

# sam model define
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# declare feature containers 
feature_list = []
feature_map_list = []

# load the mask image and convert it to a binary mask
mask = Image.open(mask_path)
mask = TF.to_tensor(mask)
origin_mask = deepcopy(mask)
unique_mask_value = torch.unique(mask)
for mv in unique_mask_value[1:-1]:
    mask = torch.where(origin_mask==mv, mv, 0)
    mask = mask.gt(0).float()  # convert to binary mask

    # get the bounding box of the mask
    nonzero = torch.nonzero(mask)
    bbox = torch.tensor([
        torch.min(nonzero[:, 1]),  # min x
        torch.min(nonzero[:, 2]),  # min y
        torch.max(nonzero[:, 1]),  # max x
        torch.max(nonzero[:, 2]),  # max y
    ])

    # expand the bounding box by a factor of `scale`
    scale = 1.2
    bbox_center = (bbox[:2] + bbox[2:]) / 2  # center point of the bounding box
    bbox_size = (bbox[2:] - bbox[:2]) * scale  # scaled size of the bounding box
    bbox = torch.cat([bbox_center - bbox_size / 2, bbox_center + bbox_size / 2])
    bbox = bbox.floor().int()

    # crop the image using the expanded bounding box
    img = Image.open(image_path)
    left, top, right, bottom = bbox.tolist()
    width, height = right - left, bottom - top
    cropped = TF.crop(img, left, top, width, height)
    cropped = np.array(cropped)
    img = TF.to_tensor(np.array(img))

    # crop mask 
    mask = TF.crop(mask, left, top, width, height)

    # pass the image through the model to get the features
    predictor.set_image(cropped)
    features = predictor.features
    features = features.detach().cpu()
    features = F.interpolate(features, size=mask.shape[-2:], mode='nearest').squeeze(0)

    # resize the mask to the size of the features and perform element-wise multiplication
    # mask = TF.resize(mask.unsqueeze(0), features.shape[-2:], Image.NEAREST).squeeze()
    mask_features = features * mask
    mask_features = torch.mean(mask_features, dim=1, keepdim=True) # First Average Pooling then image resize. Heuristic
    mask_features = F.interpolate(mask_features.unsqueeze(0), size=img.shape[-2:], mode='nearest').squeeze(0)

    # average pooling on mask 
    mask_features = mask_features.squeeze(1)

    # save mask features in feature list
    feature_list.append(mask_features.flatten().numpy())
    feature_map_list.append(np.where(mask_features.numpy()!=0, 1, 0))

# calculate similarity between mask 
feature_list = np.array(feature_list)
sim_matrix = cosine_similarity(feature_list, feature_list)
print(sim_matrix)

# Set labels
class_labels = [1, 1, 1, 2]

# mask coloring 
mask = deepcopy(origin_mask)
colored_mask = torch.cat((mask, mask, mask))
for mv, lb in zip(unique_mask_value[1:-1], class_labels):
    _, x_index, y_index = torch.where(origin_mask==mv)
    for x, y in zip(x_index, y_index):
        if lb == 0: 
            channel = 0
        elif lb == 1:
            channel = 1
        elif lb == 2:
            channel = 2
        colored_mask[channel, x.item(), y.item()] = 255 

# Save image
colored_mask = np.transpose(colored_mask.cpu().numpy(), (1, 2, 0))  # (C, H, W) -> (H, W, C)

# create PIL image object from numpy array and save to file
img_pil = Image.fromarray(np.uint8(colored_mask))
img_pil.save(f"/home/injo/solutions/mars/data/2007_{image_index.zfill(6)}.jpg")

print("here")