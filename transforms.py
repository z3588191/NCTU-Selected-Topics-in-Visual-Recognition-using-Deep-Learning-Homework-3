import torch
import random
import math

from torchvision.transforms import functional as F
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            image = F.hflip(image)

            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

            masks = target["masks"]
            masks = F.hflip(masks)
            target["masks"] = masks

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __call__(self, image, target):
        image = F.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return image, target


class photometric_distort(object):
    def __call__(self, image, target):
        distortions = [F.adjust_brightness,
                       F.adjust_contrast,
                       F.adjust_saturation,
                       F.adjust_gamma,
                       F.adjust_hue]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ is 'adjust_hue':
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    adjust_factor = random.uniform(0.5, 1.5)
                # Apply this distortion
                image = d(image, adjust_factor)

        return image, target


class Resize(object):
    def __init__(self, size=513):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, image, target):
        width, height = image.size
        w_scale = self.size[0] / width
        h_scale = self.size[1] / height
        boxes = target['boxes']
        boxes[:, 0] = boxes[:, 0] * w_scale
        boxes[:, 2] = boxes[:, 2] * w_scale
        boxes[:, 1] = boxes[:, 1] * h_scale
        boxes[:, 3] = boxes[:, 3] * h_scale
        target['boxes'] = boxes

        image = F.resize(image, self.size, Image.BILINEAR)
        target['masks'] = F.resize(target['masks'], self.size, Image.NEAREST)

        return image, target


def get_corners(boxes):
    width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)

    x1 = boxes[:, 0].view(-1, 1)
    y1 = boxes[:, 1].view(-1, 1)
    x2 = x1 + width
    y2 = y1
    x3 = x1
    y3 = y1 + height
    x4 = boxes[:, 2].view(-1, 1)
    y4 = boxes[:, 3].view(-1, 1)
    corners = torch.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def change_origin2center(boxes, width, height):
    boxes[:, 0] = boxes[:, 0] - width // 2
    boxes[:, 2] = boxes[:, 2] - width // 2
    boxes[:, 1] = boxes[:, 1] - height // 2
    boxes[:, 3] = boxes[:, 3] - height // 2

    return boxes


def boxes_rotate(boxes, angle, ori_w, ori_h, new_w, new_h):
    # change coord. base to image center
    boxes = change_origin2center(boxes, ori_w, ori_h)

    # get 4 corners of boxes
    corners = get_corners(boxes)

    cos = math.cos(angle / 180 * math.pi)
    sin = math.sin(angle / 180 * math.pi)
    Rotation_matrix = torch.tensor([[cos, -sin],
                                    [sin, cos]])

    # rotate the corner coord.
    new_corners = corners.to(torch.float)
    new_corners[:, :2] = torch.matmul(new_corners[:, :2], Rotation_matrix)
    new_corners[:, 2:4] = torch.matmul(new_corners[:, 2:4], Rotation_matrix)
    new_corners[:, 4:6] = torch.matmul(new_corners[:, 4:6], Rotation_matrix)
    new_corners[:, 6:] = torch.matmul(new_corners[:, 6:], Rotation_matrix)
    new_corners = torch.round(new_corners).to(torch.long)

    # find correct boxes in new expanded-rotation img
    new_left1 = torch.minimum(new_corners[:, 0], new_corners[:, 4]).view(-1, 1)  # min(x1, x3)
    new_top1 = torch.minimum(new_corners[:, 1], new_corners[:, 3]).view(-1, 1)  # min(y1, y2)
    new_right1 = torch.maximum(new_corners[:, 2], new_corners[:, 6]).view(-1, 1)  # max(x2, x4)
    new_bottom1 = torch.maximum(new_corners[:, 5], new_corners[:, 7]).view(-1, 1)  # min(y3, y4)
    new_left2 = ((new_corners[:, 0] + new_corners[:, 4]) // 2).view(-1, 1)  # (x1 + x3) / 2
    new_top2 = ((new_corners[:, 1] + new_corners[:, 3]) // 2).view(-1, 1)  # (y1 + y2) / 2
    new_right2 = ((new_corners[:, 2] + new_corners[:, 6]) // 2).view(-1, 1)  # (x2 + x4) / 2
    new_bottom2 = ((new_corners[:, 5] + new_corners[:, 7]) // 2).view(-1, 1)  # (y3 + y4) / 2

    new_left = (new_left1 + new_left2) // 2
    new_top = (new_top1 + new_top2) // 2
    new_right = (new_right1 + new_right2) // 2
    new_bottom = (new_bottom1 + new_bottom2) // 2

    # change coord. base to new image left-top
    new_left = new_left + new_w // 2
    new_right = new_right + new_w // 2
    new_top = new_top + new_h // 2
    new_bottom = new_bottom + new_h // 2

    new_boxes = torch.hstack((new_left, new_top, new_right, new_bottom))

    return new_boxes


class RandomRotate(object):
    def __init__(self, angle=(-15, 15)):
        # angle must be less than 45
        self.angle = angle

    def __call__(self, image, target):
        ang = random.uniform(self.angle[0], self.angle[1])

        ori_w, ori_h = image.size
        image = F.rotate(image, angle=ang, resample=Image.BILINEAR, expand=True,
                         fill=(int(0.485*255), int(0.456*255), int(0.406*255)))
        new_w, new_h = image.size

        target['masks'] = F.rotate(target['masks'], angle=ang, resample=Image.NEAREST, expand=True)

        target['boxes'] = boxes_rotate(target['boxes'], ang, ori_w, ori_h, new_w, new_h)

        return image, target


def find_intersection(set_1, set_2):
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


class Random_crop(object):
    def __call__(self, image, target):
        original_w, original_h = image.size
        image = F.to_tensor(image)
        boxes = target['boxes']
        labels = target['labels']
        masks = target['masks']

        # Keep choosing a minimum overlap until a successful crop is made
        min_overlap = 0.75

        # Try up to 50 times for this choice of minimum overlap
        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.75
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.LongTensor([left, top, right, bottom])

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find boxes in cropped region
            boxes_in_crop = (boxes[:, 0] < right) * (boxes[:, 2] > left) * (boxes[:, 1] < bottom) * (boxes[:, 3] > top)
            if not boxes_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[boxes_in_crop, :]
            new_masks = masks[boxes_in_crop, :]
            new_labels = labels[boxes_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            # Crop masks
            new_masks = new_masks[:, top:bottom, left:right]

            new_target = {}
            new_target['boxes'] = new_boxes
            new_target['labels'] = new_labels
            new_target['masks'] = new_masks
            new_target['image_name'] = target['image_name']

            new_image = F.to_pil_image(new_image)
            return new_image, new_target

        image = F.to_pil_image(image)
        return image, target
