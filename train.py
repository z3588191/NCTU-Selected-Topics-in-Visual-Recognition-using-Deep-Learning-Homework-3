import time
import torch
from dataset import PascalDataset
import torchvision
from model import *
from utils import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# hyperparameters
min_size = [256, 384, 512]
max_size = 800
num_classes = 21
batch_size = 8
epochs = 100
warmup_epochs = 5
workers = 4
print_freq = 50
lr = 0.007
milestones = [40, 45]
gamma = 0.1
momentum = 0.9
weight_decay = 5e-4
device = "cuda:0"
torch.cuda.set_device(0)


# dataloader
train_dataset = PascalDataset(is_train=True, split=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers)

valid_dataset = PascalDataset(is_train=False, split=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=valid_dataset.collate_fn, num_workers=workers)


# model
backbone = ResNestWithFPN()
model = torchvision.models.detection.MaskRCNN(backbone, num_classes, min_size=min_size, max_size=max_size)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 512
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model = model.to(device)
lrx1_params = []
lrx10_params = []
for n, p in model.named_parameters():
    if p.requires_grad:
        if n.startswith('backbone.body'):
            lrx1_params.append(p)
        else:
            lrx1_params.append(p)

optimizer = torch.optim.SGD(
                [{'params': lrx1_params, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
                 {'params': lrx10_params, 'lr': lr*10, 'momentum': momentum, 'weight_decay': weight_decay}])
scheduler = Poly_LR_Scheduler(lr, epochs, len(train_dataloader), warmup_epochs=warmup_epochs)


# Training
best_mAP = 0.
best_loss = 10000.

for epoch in range(1, epochs+1):
    loss_epoch = 0.
    start = time.time()
    loss_value = 0
    model.train()
    for i, (images, targets) in enumerate(train_dataloader):
        # Move to default device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'masks': t['masks'].to(device),
                    'labels': t['labels'].to(device)} for t in targets]

        optimizer.zero_grad()
        scheduler(optimizer, i, epoch-1)
        # Forward prop.
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value += losses.item()
        loss_epoch += losses.item()

        losses.backward()
        optimizer.step()

        # Print status
        if (i+1) % print_freq == 0:
            end = time.time()
            print('Epoch: [{:>6d}][{:>6d}/{:>6d}]\tTime: {:.3f}\t'
                  'Loss: {:.4f}\t'.format(epoch, (i+1)*batch_size,
                                          len(train_dataset), end-start, loss_value / print_freq))
            start = time.time()
            loss_value = 0
    print('Epoch: {:>2d}, Loss: {:.4f}\t'.format(epoch, loss_epoch / len(train_dataset) * batch_size))

    # Get Validation loss
    valid_loss = 0.
    with torch.no_grad():
        for i, (images, targets) in enumerate(valid_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{'boxes': t['boxes'].to(device), 'masks': t['masks'].to(device),
                        'labels': t['labels'].to(device)} for t in targets]
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            valid_loss += losses.item()

    # Get Validation mAP
    model.eval()
    det_masks = list()
    det_labels = list()
    det_scores = list()
    true_masks = list()
    true_labels = list()
    with torch.no_grad():
        for i, (images, targets) in enumerate(valid_dataloader):
            images = list(image.to(device) for image in images)
            masks = [t['masks'].to(device) for t in targets]
            labels = [t['labels'].to(device) for t in targets]

            outputs = model(images)
            out_masks = [o['masks'].squeeze(1).round() for o in outputs]
            out_labels = [o['labels'] for o in outputs]
            out_scores = [o['scores'] for o in outputs]

            true_masks.extend(masks)
            true_labels.extend(labels)
            det_masks.extend(out_masks)
            det_labels.extend(out_labels)
            det_scores.extend(out_scores)

    mAP = calculate_AP(det_masks, det_labels, det_scores, true_masks, true_labels, 0.5, device)

    print("valid Loss: {:.4f}, valid mAP: {:.4f}".format(valid_loss/len(valid_dataset)*batch_size, mAP))
    if valid_loss < best_loss:
        torch.save(model, "maskRcnn_bestloss.pth")
        best_loss = valid_loss

    if mAP > best_mAP:
        torch.save(model, "maskRcnn_bestmAP.pth")
        best_mAP = mAP
