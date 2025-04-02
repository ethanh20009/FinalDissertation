from typing import cast
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation
from PIL import Image
import cv2
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
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
from utils_live import check_and_create_dir
import numpy as np
import os.path as osp
import PIL
import PIL.Image


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    print("Loading Dataset...")
    ds = cast(
        Dataset,
        load_dataset("timm/mini-imagenet", split="train")
        .shuffle(seed=2)
        .with_format("numpy"),
    )

    num_generations = 5
    for image_index in range(num_generations):
        ds_image = ds[int(image_index)]["image"]
        save_target_img(ds_image)
        # layered_vectorisation(ds_image)


def save_target_img(gt):
    filename = str(uuid.uuid4())
    target_path = osp.join("composed_targets", filename + ".png")
    check_and_create_dir(target_path)
    PIL.Image.fromarray(gt).save(target_path, "PNG")


def layered_vectorisation(gt: NDArray):
    """
    Perform layered vectorisation on the input image.
    Args:
        gt (numpy.ndarray): The input image.
    """
    print("Performing depth estimation...")
    depth = perform_depth_estimation(gt)
    # plt.imshow(depth)
    # plt.show()

    seg_masks, num_masks, seg_info = perform_segmentation(gt)

    seg_info_dict = {"0": {"id": 0, "isthing": False}}
    for info in seg_info:
        seg_info_dict[str(info["id"])] = info

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
    # ax[0].imshow(gt)
    # ax[1].imshow(seg_masks)
    # plt.show()

    # Show image and mask as greyscale
    # plt.imshow(gt)
    # plt.imshow(np.where(mask_list[0]["mask"] == 1, 0, 1), cmap="gray")
    # plt.show()

    layers = []

    h = gt.shape[0]
    w = gt.shape[1]

    bg_inpainted = inpaint_bg(gt, mask_list[0]["mask"])

    # Show inpaint
    # plt.imshow(np.asarray(bg_inpainted))
    # plt.show()

    # Fit svg layer to each mask.
    for i in range(0, num_masks + 1):
        target = gt if i > 0 else np.asarray(bg_inpainted)
        mask = mask_list[i]["mask"] if i > 0 else None
        generation_method = (
            "experiment_exp2_64"
            if seg_info_dict[str(i)]["isthing"]
            else "experiment_exp2_32"
        )
        new_shapes, new_shape_groups = live_run(
            target,
            ["TestVectorisation"],
            experiment=generation_method,
            mask=mask,
        )
        layers.append((new_shapes, new_shape_groups))

    # Make UUID to save svg
    save_svg_composed("composed_images/" + str(uuid.uuid4()) + ".svg", w, h, layers)


def perform_segmentation(test_image):
    print("Running segmentation")
    seg_args = u2seg_demo.get_parser().parse_args()
    seg_cfg = u2seg_demo.setup_cfg(seg_args)

    demo = VisualizationDemo(seg_cfg)
    seg_predictions, seg_viz = demo.run_on_image(test_image)
    # instances: Instances = seg_predictions["instances"]
    segmentation_masks: torch.FloatTensor = seg_predictions["panoptic_seg"][0]
    segmentation_masks_info: torch.FloatTensor = seg_predictions["panoptic_seg"][1]
    num_masks = len(segmentation_masks_info)
    seg_masks = segmentation_masks.cpu().numpy()

    # plt.imshow(seg_viz.get_image())
    # plt.show()
    #
    # plt.imshow(np.where(seg_masks == 1, 1, 0))

    return seg_masks, num_masks, segmentation_masks_info


def perform_depth_estimation(test_image):
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


def inpaint_bg(bg, mask):
    image = Image.fromarray(bg.astype("uint8")).convert("RGB")
    inv_mask = np.where(mask == 0, 255, 0)
    inv_mask = dilate_mask(inv_mask)
    mask_image = Image.fromarray(inv_mask.astype("uint8")).convert("RGB")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    original_width, original_height = image.size

    width = (original_width // 8) * 8
    height = (original_height // 8) * 8

    image = image.resize((width, height))
    mask_image = mask_image.resize((width, height))

    image_result = pipe(
        image=image,
        mask_image=mask_image,
        prompt="background",
        height=height,
        width=width,
        guidance_scale=1.0,
    ).images[0]
    inpainted_image = image_result.resize((original_width, original_height))
    return inpainted_image


def dilate_mask(mask_array, dilation_iterations=5):
    binary_mask = mask_array > 0
    dilated_mask = binary_dilation(binary_mask, iterations=dilation_iterations)

    # Convert back and return
    return dilated_mask.astype(np.uint8) * 255


if __name__ == "__main__":
    main()
