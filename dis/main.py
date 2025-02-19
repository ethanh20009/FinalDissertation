from typing import cast
from datasets import Dataset, load_dataset
import sys
import os

# Get the absolute path to the 'dep' directory
dep_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Useg"))

# Add the 'dep' directory to sys.path
sys.path.append(dep_dir)

import predictor, u2seg_demo
from u2seg_demo import VisualizationDemo


def main():
    ds = cast(
        Dataset,
        load_dataset("Sourabh2/Emoji_cartoon", split="train").with_format("numpy"),
    )
    # Show image
    print(ds[0])

    seg_args = u2seg_demo.get_parser().parse_args()
    seg_cfg = u2seg_demo.setup_cfg(seg_args)

    demo = VisualizationDemo(seg_cfg)


if __name__ == "__main__":
    main()
