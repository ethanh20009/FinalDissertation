from typing import cast
from datasets import Dataset, load_dataset


def main():
    ds = cast(
        Dataset,
        load_dataset("Sourabh2/Emoji_cartoon", split="train").with_format("torch"),
    )
    print(type(ds))

    # Show image
    print(ds[0])


if __name__ == "__main__":
    main()
