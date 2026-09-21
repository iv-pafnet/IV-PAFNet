"""Paired visible/infrared COCO dataset for IV-PAFNet."""

import os
import copy
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
from PIL import Image

from src.core import register

__all__ = ['CocoDetection']

@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, img_folder_ir=None, transforms=None, return_masks=False, remap_mscoco_category=False):
        # torchvision stores the visible-image directory in self.root.
        super(CocoDetection, self).__init__(img_folder, ann_file)

        if img_folder_ir is None:
            raise ValueError("img_folder_ir is required for paired RGB/IR training")
        self.img_folder_ir = img_folder_ir
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        # Load the image identifier and annotations through the COCO API.
        image_id = self.ids[idx]
        target = self._load_target(image_id)
        target = {'image_id': image_id, 'annotations': target}

        # Use the same COCO file name for the paired visible and infrared images.
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        rgb_path = os.path.join(self.root, file_name)
        img_rgb = Image.open(rgb_path).convert("RGB")
        
        ir_path = os.path.join(self.img_folder_ir, file_name)
        img_ir = Image.open(ir_path).convert("RGB")
        if img_rgb.size != img_ir.size:
            raise ValueError(
                f"paired image size mismatch for {file_name}: "
                f"visible={img_rgb.size}, infrared={img_ir.size}"
            )

        # Convert COCO annotations before applying paired transforms.
        img_rgb, img_ir, target = self.prepare(img_rgb, img_ir, target)

        if 'boxes' in target:
            target['boxes'] = tv_tensors.BoundingBoxes(
                target['boxes'],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_rgb.size[::-1])  # h w

        if 'masks' in target:
            target['masks'] = tv_tensors.Mask(target['masks'])

        # Replay the same random transform parameters for both modalities.
        if self._transforms is not None:
            torch_state = torch.get_rng_state()
            random_state = random.getstate()
            np_state = np.random.get_state()
            
            # Clone tensor subclasses explicitly before replaying transforms.
            target_copy = {}
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    target_copy[k] = v.clone()
                else:
                    target_copy[k] = copy.deepcopy(v)

            img_rgb, target_rgb = self._transforms(img_rgb, target)

            # Restore RNG state so the infrared branch receives the same transform.
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            np.random.set_state(np_state)

            img_ir, _ = self._transforms(img_ir, target_copy)

            target_final = target_rgb
        else:
            target_final = target

        # Return a six-channel tensor: visible RGB followed by infrared RGB.
        return torch.cat([img_rgb, img_ir], dim=0), target_final

    def extra_repr(self) -> str:
        s = f' img_folder: {self.root}\n ann_file: {self.ann_file}\n'
        if hasattr(self, 'img_folder_ir'):
            s += f' img_folder_ir: {self.img_folder_ir}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s

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
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image_rgb, image_ir, target):
        w, h = image_rgb.size

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

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

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
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
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
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image_rgb, image_ir, target

# DroneVehicle category mapping.
mscoco_category2name = {
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'van',
    5: 'freight_car'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
