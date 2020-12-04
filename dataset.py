import random
from operator import itemgetter
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from transforms import *


class PascalDataset(Dataset):
    def __init__(self, is_train=True, split=False):
        self.is_train = is_train
        self.split = split

        if is_train:
            transforms = [
                photometric_distort(),
                Random_crop(),
                RandomHorizontalFlip(0.5),
                RandomRotate(angle=(-20, 20)),
                ToTensor(), ]
            self.transforms = Compose(transforms)
        else:
            transforms = [
                ToTensor(), ]
            self.transforms = Compose(transforms)

        coco = COCO("pascal_train.json")
        self.img_names = []
        self.labels = []
        self.boxes = []
        self.masks = []
        for key in coco.imgs.keys():
            img_info = coco.loadImgs(ids=key)
            self.img_names.append(img_info[0]['file_name'])

            annids = coco.getAnnIds(imgIds=key)
            anns = coco.loadAnns(annids)
            label = np.zeros(len(anns))
            box = np.zeros((len(anns), 4))
            mask = np.zeros((len(anns), img_info[0]['height'], img_info[0]['width']))
            for i in range(len(anns)):
                label[i] = anns[i]['category_id']
                box[i] = anns[i]['bbox']
                mask[i] = coco.annToMask(anns[i])

            box[:, 2] = box[:, 0] + box[:, 2]
            box[:, 3] = box[:, 1] + box[:, 3]

            self.labels.append(label)
            self.boxes.append(box)
            self.masks.append(mask)

        if split:
            idx_shuffle = np.random.permutation(len(self.img_names))
            train_idx = idx_shuffle[:-50]
            valid_idx = idx_shuffle[-50:]
            if is_train:
                self.img_names = list(itemgetter(*train_idx)(self.img_names))
                self.labels = list(itemgetter(*train_idx)(self.labels))
                self.boxes = list(itemgetter(*train_idx)(self.boxes))
                self.masks = list(itemgetter(*train_idx)(self.masks))
            else:
                self.img_names = list(itemgetter(*valid_idx)(self.img_names))
                self.labels = list(itemgetter(*valid_idx)(self.labels))
                self.boxes = list(itemgetter(*valid_idx)(self.boxes))
                self.masks = list(itemgetter(*valid_idx)(self.masks))

    def __getitem__(self, i):
        # Read image
        img = Image.open("train_images/" + self.img_names[i]).convert('RGB')

        #  get category_id of mask
        labels = torch.LongTensor(self.labels[i])  # (n_objects)

        #  get bbox of mask
        boxes = torch.LongTensor(self.boxes[i])  # (n_objects, 4)

        #  get mask
        masks = torch.LongTensor(self.masks[i])  # (n_objects, height, width)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_name"] = self.img_names[i]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_names)

    def collate_fn(self, batch):
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        return images, targets
