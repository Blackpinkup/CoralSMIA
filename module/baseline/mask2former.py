import torch
import torch.nn as nn
import numpy as np
from ever.interface import ERModule
from ever import registry
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

def pre_process(semantic_mask: torch.Tensor, ignore_index=-1):
    gt_masks_batch = []
    gt_classes_batch = []
    # semantic_mask [bs x H x W]
    for semantic_mask_i in semantic_mask:
        classes = torch.unique(
            semantic_mask_i,
            sorted=False,
            return_inverse=False,
            return_counts=False)
        # remove ignored region
        gt_labels = classes[classes != ignore_index]

        masks = []
        for class_id in gt_labels:
            masks.append((semantic_mask_i == class_id))

        if len(masks) == 0:
            gt_masks = torch.zeros(
                (0, semantic_mask_i.shape[-2],
                 semantic_mask_i.shape[-1])).to(semantic_mask_i)
        else:
            gt_masks = torch.stack(masks)
        gt_masks_batch.append(gt_masks.float())
        gt_classes_batch.append(gt_labels.long())

    return gt_masks_batch, gt_classes_batch



@registry.MODEL.register()
class Mask2Former(ERModule):
    def __init__(self, config):
        super(Mask2Former, self).__init__(config)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.pretrain_dir)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(self.config.pretrain_dir, num_labels=8, num_queries=8, ignore_mismatched_sizes=True)


    def forward(self, x, y=None):
        if self.training:
            #mask = y['mask']
            mask = y['cls']
            gt_masks_batch, gt_classes_batch =  pre_process(mask)
            outputs = self.mask2former(pixel_values=x, mask_labels=gt_masks_batch, class_labels=gt_classes_batch)
            #print(outputs.class_queries_logits.shape)
            #print(outputs.masks_queries_logits.shape)
            return dict(loss=outputs.loss)
        else:
            outputs = self.mask2former(pixel_values=x)
            target_sizes = [x_i.shape[-2:] for x_i in x]
            pred_semantic_map = self.image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            
            return torch.stack(pred_semantic_map)


    def set_default_config(self):
        self.config.update(dict(
            pretrain_dir='/home/biqi/code/business/LoveNAS-master/pretrain/mask2former-swin-base-IN21k-cityscapes-semantic/',
        ))

if __name__ == '__main__':
    from skimage.io import imread
    image = torch.ones(1, 3, 1024, 1024).to(device)
    y =dict(
        mask = torch.ones(1, 1024, 1024).to(device)
    )
    m = Mask2Former(dict()).to(device)
    l = m(image, y)
