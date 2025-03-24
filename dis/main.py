from typing import cast
import cv2
import uuid
from datasets import Dataset, load_dataset
import sys
import os
from detectron2.structures.instances import Instances
from matplotlib import pyplot as plt
import torch
from depth_anything_v2.dpt import DepthAnythingV2

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
        load_dataset("timm/mini-imagenet", split="train").with_format("numpy"),
    )
    # Show image
    test_image = ds[3]["image"]

    print("Performing depth estimation...")
    depth = perform_depth_estimation(test_image)
    # plt.imshow(depth)
    # plt.show()

    seg_masks, num_masks, seg_info = perform_segmentation(test_image)

    # Split single integer tagged image mask into array of binary mask objects
    mask_list = [
        {"mask": np.where(seg_masks == i, 1, 0), "id": i}
        for i in range(0, num_masks + 1)
    ]

    # Find average depth and add to mask object
    for mask_obj in mask_list:
        mask_obj["depth"] = np.mean(depth[mask_obj["mask"] == 1])

    # Sort layers from furthest to closest
    mask_list = sorted(mask_list, key=lambda x: x["depth"])

    # Find background by largest area that is not 'isthing'
    bg_ids = filter(lambda x: not bool(x["isthing"]), seg_info)
    max_bg = max(bg_ids, key=lambda x: x["area"])

    # Move the "best" background to the front
    for i in range(0, len(mask_list)):
        if mask_list[i]["id"] == max_bg["id"]:
            mask_list.insert(0, mask_list.pop(i))

    # Plot ordered masks with subplots
    # fig, ax = plt.subplots(num_masks + 1)
    #
    # for i in range(0, num_masks + 1):
    #     ax[i].imshow(mask_list[i]["mask"])
    #     ax[i].set_title(
    #         "ID: " + str(mask_list[i]["id"]) + ", Depth: " + str(mask_list[i]["depth"])
    #     )
    # plt.show()

    # Plot original image and segmentation masks
    # fig, ax = plt.subplots(3)
    # ax[0].imshow(test_image)
    # ax[1].imshow(seg_masks)
    # plt.show()

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


def perform_segmentation(test_image):
    print("Running segmentation")
    seg_args = u2seg_demo.get_parser().parse_args()
    seg_cfg = u2seg_demo.setup_cfg(seg_args)

    demo = VisualizationDemo(seg_cfg)
    seg_predictions, seg_viz = demo.run_on_image(test_image)
    # instances: Instances = seg_predictions["instances"]
    segmentation_masks: torch.FloatTensor = seg_predictions["panoptic_seg"][0]
    segmentation_masks_info: torch.FloatTensor = seg_predictions["panoptic_seg"][1]
    print(segmentation_masks_info)
    num_masks = len(segmentation_masks_info)
    seg_masks = segmentation_masks.cpu().numpy()

    # plt.imshow(seg_viz.get_image())
    # plt.show()
    #
    # plt.imshow(np.where(seg_masks == 1, 1, 0))

    return seg_masks, num_masks, segmentation_masks_info


def perform_depth_estimation(test_image):
    print(test_image)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DepthAnythingV2(
        encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
    )
    model.load_state_dict(
        torch.load("ckpts/depth_anything_v2_vitl.pth", map_location="cuda")
    )
    model.to(device)
    model.eval()

    depth = model.infer_image(np.array(test_image))
    return depth


if __name__ == "__main__":
    main()
