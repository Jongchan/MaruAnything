import os

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import show_anns


class Segment:

    def __init__(self, image_id, segment_idx, binary_mask, roi_feature):
        self.image_id = image_id
        self.segment_idx = segment_idx
        self.binary_mask = binary_mask
        self.roi_feature = roi_feature

    def save_to(self, save_dir: str):
        save_path = f"{save_dir}/{self.image_id}/{self.segment_idx}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dict = {
            "image_id": self.image_id,
            "segment_idx": self.segment_idx,
            "binary_mask": self.binary_mask,
            "roi_feature": self.roi_feature,
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

class Segmenter:

    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str = "cuda"
    ):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        # we can also finetune the segmentation results by tuning via options:
        # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb  # noqa
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print(f"Segmenter initialized with {sam_checkpoint} and {model_type}")

    def get_segments(self, image_id, image_path: str, save_vis: bool = False):
        assert os.path.isfile(image_path), f"Image path {image_path} does not exist"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks, features_and_boxes = self.mask_generator.generate(image)
        if save_vis:
            self._save_vis(image, masks, image_id)
        
        # TODO: there may be multiple boxes in the future, but for now we assume only one crop
        assert len(features_and_boxes) == 1, "Only support one object per image"
        feature_img, _ = features_and_boxes[0]
        feature_img = feature_img.cpu().numpy()

        segments = []
        for mask_idx, mask in enumerate(masks):
            roi_feature = get_roi_feature(feature_img, mask)
            segment = Segment(image_id, mask_idx, mask["segmentation"], roi_feature)
            segments.append(segment)
        
        return segments

    def _save_vis(self, image, masks, image_id, vis_dir: str = "./vis"):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')

        vis_path = os.path.join(vis_dir, f"{image_id}.png")
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(vis_path)
        plt.close()

def get_roi_feature(feature_img, roi_mask, eps=1e-8):
    mask_segment = roi_mask["segmentation"].astype(np.float32)
    mask_segment_resized = cv2.resize(mask_segment, feature_img.shape[-2:], interpolation=cv2.INTER_LINEAR)
    mask_segment_resized = np.expand_dims(mask_segment_resized, axis=(0, 1))
    # TODO: check if indexing is correct. i.e. the mask may be inverted / transposed.
    roi_feature = np.sum(feature_img * mask_segment_resized, axis=(0, 2, 3)) / (np.sum(mask_segment_resized) + eps)
    return roi_feature
