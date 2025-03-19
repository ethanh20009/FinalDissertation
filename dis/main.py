from typing import cast
from datasets import Dataset, load_dataset
import sys
import os
from detectron2.structures.instances import Instances
from matplotlib import pyplot as plt
import torch

# Get the absolute path to the 'dep' directory
dep_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Useg"))

# Add the 'dep' directory to sys.path
sys.path.append(dep_dir)

from main_live import live_run
import predictor, u2seg_demo
from u2seg_demo import VisualizationDemo
import numpy as np


def main():
    print("Loading Dataset...")
    ds = cast(
        Dataset,
        load_dataset("Sourabh2/Emoji_cartoon", split="train").with_format("numpy"),
    )
    # Show image
    test_image = ds[3]["image"]

    seg_args = u2seg_demo.get_parser().parse_args()
    seg_cfg = u2seg_demo.setup_cfg(seg_args)

    demo = VisualizationDemo(seg_cfg)
    print("Running segmentation")
    seg_predictions, seg_viz = demo.run_on_image(test_image)
    print("Seg Predictions")
    instances: Instances = seg_predictions["instances"]
    print(instances.get_fields()["pred_masks"].cpu().numpy().shape)
    print("Seg Viz")
    print(seg_viz)
    segmentation_masks: torch.FloatTensor = seg_predictions["panoptic_seg"][0]
    seg_masks = segmentation_masks.cpu().numpy()
    seg_mask_layer = np.where(seg_masks == 1, 1, 0)

    fig, ax = plt.subplots(3)
    ax[0].imshow(test_image)
    ax[1].imshow(seg_masks)
    ax[2].imshow(seg_mask_layer)
    print(seg_mask_layer)
    plt.show()

    live_run(
        test_image, ["smile"], experiment="experiment_exp2_32", mask=seg_mask_layer
    )


if __name__ == "__main__":
    main()
