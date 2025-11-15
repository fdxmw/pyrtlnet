import argparse
import pathlib
import shutil
import sys

import numpy as np

from pyrtlnet.cli_util import display_image, display_outputs
from pyrtlnet.constants import quantized_model_prefix
from pyrtlnet.inference_util import (
    add_common_arguments,
    batched_images,
    load_mnist_data,
)
from pyrtlnet.litert_inference import load_tflite_model, run_tflite_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="litert_inference.py")
    add_common_arguments(parser)
    args = parser.parse_args()

    assert args.batch_size == 1

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST test data.
    test_images, test_labels = load_mnist_data(args.tensor_path)

    tflite_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.tflite"
    if not tflite_file.exists():
        sys.exit(f"{tflite_file} not found. Run tensorflow_training.py first.")
    interpreter = load_tflite_model(quantized_model_name=tflite_file)

    correct = 0
    for batch_start_index, test_batch in batched_images(
        test_images, args.start_image, args.num_images, args.batch_size
    ):
        # Print the test image.
        test_image = test_batch[0]
        print(f"LiteRT network input (#{batch_start_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        # Run LiteRT inference on the test image.
        layer0_output, layer1_output, actual = run_tflite_model(
            interpreter=interpreter, test_image=test_image
        )

        # Print results.
        print(f"LiteRT layer 0 output {layer0_output.shape} {layer0_output.dtype}")
        print(f"{layer0_output}\n")
        print(f"LiteRT layer 1 output {layer1_output.shape} {layer1_output.dtype}")
        print(f"{layer1_output}\n")

        expected = test_labels[batch_start_index]
        print(f"LiteRT network output (#{batch_start_index}):")
        display_outputs(layer1_output.flatten(), expected, actual)
        if actual == expected:
            correct += 1

        if batch_start_index < args.num_images - 1:
            print()

    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
