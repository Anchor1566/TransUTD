# Modified by zbx
# ------------------------------------------------------------------------
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T

from PIL import Image 
from pycocotools import mask as coco_mask

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

import os
import random

from .coco_video_parser import CocoVID

__all__ = ['CocoDetection']


@register()
class CocoVideoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, 
                 img_folder, 
                 ann_file, 
                 transforms, 
                 num_refs=3, 
                 return_masks=False, 
                 is_train=True,  # FIXME: Hack implementation
                 filter_key_img=True,
                 remap_mscoco_category=False):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        
        self.num_refs = num_refs
        self.is_train = is_train 
        # self.is_train = True if 'train' in ann_file else False  # FIXME: Hack implementation
        self.filter_key_img = filter_key_img
        # self.transform_ref = transform_ref
        self.cocovid = CocoVID(ann_file)
        self.test_with_one_img = False

    def __getitem__(self, idx):
        images, target = self.load_item(idx)
        if self._transforms is not None:
            images, target, _ = self._transforms(images, target, self)
        
        # [C*T, H, W]
        return torch.cat(images, dim=0), target
        
    def load_item(self, idx):
        image, target = super(CocoVideoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]  # current image id
        target = {'image_id': image_id, 'annotations': target}
        
        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)
          
        target['idx'] = torch.tensor([idx])
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        # sample reference frames
        coco = self.coco
        img_info = coco.loadImgs(image_id)[0]
        video_id = img_info.get('video_id', -1)
        ref_img_ids = self.sample_ref_frames(image_id, video_id)

        images = [image]
        for ref_id in ref_img_ids:
            ref_path = coco.loadImgs(ref_id)[0]['file_name']
            ref_img = self.get_image(ref_path)
            images.append(ref_img)
            
        return images, target

    def sample_ref_frames(self, img_id, video_id):
        ref_img_ids = []
        sample_range = []
        min_offset = -(self.num_refs + 1)
        max_offset = (self.num_refs + 1)

        if video_id == -1:
            return [img_id] * (self.num_refs)
        
        img_ids = self.cocovid.get_img_ids_from_vid(video_id)
        current_idx = img_ids.index(img_id)
        if self.is_train:
            window_size = min(self.num_refs + 1, len(img_ids))  
            
            # TODO: Hack implementation of sampling reference frames
            left = max(0, current_idx - window_size)
            right = min(len(img_ids), current_idx + window_size)
            sample_range = img_ids[left:right]
            
            if self.filter_key_img and img_id in sample_range:
                sample_range.remove(img_id)
                
            while len(sample_range) < (self.num_refs):
                sample_range.extend([img_id])
            # random sample num_refs from sample_range
            # TODO: add a sampling method to sample local and global frames
            ref_img_ids = random.sample(sample_range, self.num_refs)
            return ref_img_ids
            # return [img_id] * (self.num_refs)
        

        else:
            if self.test_with_one_img:
                return [img_id] * (self.num_refs)
            
            left_indexs = img_id + min_offset
            right_indexs = img_id + max_offset
            interval = int((right_indexs - left_indexs) // (2 * (self.num_refs + 1)))

            for i in range(left_indexs, right_indexs + 1, interval):
                if i < 0:
                    index = max(img_id + i, img_ids[0])
                    sample_range.append(index)
                elif i > 0:
                    index = min(img_id + i, img_ids[-1])
                    sample_range.append(index)
            sample_range = list(set(sample_range))
            if self.filter_key_img and img_id in sample_range:
                sample_range.remove(img_id)
            while len(sample_range) < (self.num_refs):
                sample_range.extend([img_id])
            ref_img_ids = random.sample(sample_range, self.num_refs)
            return ref_img_ids
            # return [img_id] * (self.num_refs)
        
    
    def get_image(self, path):
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    # def extra_repr(self) -> str:
    #     s = super().extra_repr()
    #     s += f'\n is_train: {self.is_train}'
    #     s += f'\n filter_key_img: {self.filter_key_img}'
        
    #     return s

    @property
    def categories(self):
        return self.coco.dataset['categories']

    @property
    def category2name(self):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
