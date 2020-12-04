import torch


max_w = 500
max_h = 500


def calculate_AP(det_masks, det_labels, det_scores, true_masks, true_labels, threshold, device):
    assert len(det_masks) == len(det_labels) == len(det_scores) == len(true_masks) == len(true_labels)
    n_classes = 21

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects)

    Masks = torch.zeros((true_images.size(0), max_h, max_w))
    num = 0
    for i in range(len(true_labels)):
        Masks[num:num+true_masks[i].size(0), :true_masks[i].size(1), :true_masks[i].size(2)] = true_masks[i]
        num += true_masks[i].size(0)

    true_masks = Masks  # (n_objects, h, w)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)

    assert true_images.size(0) == true_masks.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)

    Masks = torch.zeros((det_images.size(0), max_h, max_w))
    num = 0
    for i in range(len(det_labels)):
        Masks[num:num+det_masks[i].size(0), :det_masks[i].size(1), :det_masks[i].size(2)] = det_masks[i]
        num += det_masks[i].size(0)

    det_masks = Masks  # (n_detections, h, w)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_masks.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background), class 0 is background
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_masks = true_masks[true_labels == c]  # (n_class_objects, h, w)
        n_class_objects = true_class_images.size(0)

        # Keep track of which true objects with this class have already been 'detected'
        true_class_masks_detected = torch.zeros(n_class_objects).to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_masks = det_masks[det_labels == c]  # (n_class_detections, h, w)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_masks.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_masks = det_class_masks[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros(n_class_detections).to(device)  # (n_class_detections)
        false_positives = torch.zeros(n_class_detections).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_mask = det_class_masks[d]  # (h, w)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, and whether they have been detected before
            object_masks = true_class_masks[true_class_images == this_image]  # (n_class_objects_in_img, h, w)
            object_index = torch.LongTensor(range(n_class_objects))[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_masks.size(0) == 0:
                false_positives[d] = 1
                continue

            tp = False
            for t in range(object_masks.size(0)):
                if true_class_masks_detected[object_index[t]] == 0:
                    intersect = torch.logical_and(object_masks[t], this_detection_mask).sum()
                    union = torch.logical_or(object_masks[t], this_detection_mask).sum()
                    IoU = (intersect + 1) / (union + 1)

                    if IoU > threshold:
                        true_positives[d] = 1
                        true_class_masks_detected[object_index[t]] = 1
                        tp = True
                        break

            if tp is False:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    return mean_average_precision


class Poly_LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch=0, warmup_epochs=0):
        self.lr = base_lr
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('Epoches %i, learning rate = %.4f' % (epoch+1, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10
