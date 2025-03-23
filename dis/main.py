from typing import cast
import uuid
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
from save_svg_composed import save_svg_composed
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
    print(seg_predictions)
    instances: Instances = seg_predictions["instances"]
    print(instances.get_fields()["pred_masks"].cpu().numpy().shape)
    print("Seg Viz")
    print(seg_viz)
    segmentation_masks: torch.FloatTensor = seg_predictions["panoptic_seg"][0]
    segmentation_masks_info: torch.FloatTensor = seg_predictions["panoptic_seg"][1]
    num_masks = len(segmentation_masks_info)
    seg_masks = segmentation_masks.cpu().numpy()

    fig, ax = plt.subplots(3)
    ax[0].imshow(test_image)
    ax[1].imshow(seg_masks)
    plt.show()

    layers = []

    h = test_image.shape[0]
    w = test_image.shape[1]

    for i in range(0, num_masks + 1):
        mask_id = i

        new_shapes, new_shape_groups = live_run(
            test_image,
            ["smile"],
            experiment="experiment_exp2_32",
            mask_hierachy=seg_masks,
            mask_index=mask_id,
        )
        layers.append((new_shapes, new_shape_groups))

    # Make UUID to save svg
    save_svg_composed(
        "composed_images/" + str(uuid.uuid4()) + ".svg", w, h, layers[::-1]
    )


if __name__ == "__main__":
    main()
