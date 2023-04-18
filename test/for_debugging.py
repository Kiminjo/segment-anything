import numpy as np
import pandas as pd 
import timm
import torch
from torchvision import transforms, datasets
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings 

warnings.filterwarnings(action="ignore")

import sys
from pathlib import Path
sys.path.append(str(list(Path(__file__).parents)[1]))
from segment_anything import sam_model_registry, SamPredictor

def load_dataset():
    datasets.VOCSegmentation(root="/home/injo/solutions/mars/data",
                             year="2012",
                             image_set="val",
                             download=True
                             )

load_dataset()
print("load_data")

debug_src = "/home/injo/solutions/mars/workspaces/injo/segment-anything"
img_src = "/home/injo/solutions/mars/workspaces/injo/segment-anything/test/images"
image_list_dog = ["ddong_dog1", "ddong_dog2", "ddong_dog3", "ddong_dog4",
              "diff_ddong_dog", "black_ddong_dog", "sesu",
              "chiwawa", "long_chiwawa"]
image_list_car = ["car", "wheel1", "wheel2", "diff_wheel"]
image_list = image_list_dog + image_list_car
# image_list = ["wheel1", "wheel2"]

sam_checkpoint = "/home/injo/solutions/mars/data/sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:2"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
res = timm.create_model("resnet50",
                        pretrained=True)
res = res.to(device)
sam.to(device=device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

predictor = SamPredictor(sam)
sam_feature_list, res_feature_list = [], []
for idx, img_name in enumerate(image_list):
    image = cv2.imread(str(Path(img_src, f"{img_name}.png")))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)

    predictor.set_image(image)
    res_feature = res.forward_features(image_tensor.unsqueeze(0))
    res_feature = torch.mean(res_feature, dim=1, keepdim=True)
    res_feature = res_feature.squeeze(1)

    features = predictor.features
    features = torch.mean(features, dim=1, keepdim=True)
    features = features.squeeze(1)
    sam_feature_list.append(features.flatten().detach().cpu().numpy()) 
    res_feature_list.append(res_feature.flatten().detach().cpu().numpy())
    print(f"{img_name} image processed")
    predictor.reset_image()

sam_feature_list = np.array(sam_feature_list)
res_feature_list = np.array(res_feature_list)
# sam_sim_matrix = cosine_similarity(sam_feature_list, sam_feature_list)
sam_sim_matrix = euclidean_distances(sam_feature_list, sam_feature_list)
sam_sim_df = pd.DataFrame(sam_sim_matrix, columns=image_list, index=image_list)
print(sam_sim_df)
print("\n")

# res_sim_matrix = cosine_similarity(res_feature_list, res_feature_list)
res_sim_matrix = euclidean_distances(res_feature_list, res_feature_list)
res_sim_df = pd.DataFrame(res_sim_matrix, columns=image_list, index=image_list)
print(res_sim_df)
print("here")