import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil
from torchvision.transforms import functional as FT
import matplotlib.pyplot as plt
from PIL import Image
import json
from pycocotools.coco import COCO
import torchvision
from model import *
import torch

device = "cuda:0"
model = torch.load("ResNestWithFPN.pth").to(device)
model.eval()


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


cocoGt = COCO("test.json")
coco_dt = []


for imgid in cocoGt.imgs:
    image = Image.open("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name']).convert("RGB")
    image = FT.to_tensor(image).unsqueeze(0).to(device)
    output = model(image)
    masks = output[0]['masks'].to("cpu").detach().numpy()
    categories = output[0]['labels'].to("cpu").detach().numpy()
    scores = output[0]['scores'].to("cpu").detach().numpy()
    n_instances = len(scores)
    if len(categories) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid  # this imgid must be same as the key of test.json
            pred['category_id'] = int(categories[i])
            pred['segmentation'] = binary_mask_to_rle(np.round(masks[i, 0]))
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)
