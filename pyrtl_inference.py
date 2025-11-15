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
from pyrtlnet.pyrtl_inference import PyRTLInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyrtl_inference.py")
    add_common_arguments(parser)
    parser.add_argument("--verilog", action="store_true", default=False)
    parser.add_argument("--axi", action="store_true", default=False)
    parser.add_argument("--initial_delay_cycles", type=int, default=0)
    args = parser.parse_args()

    assert args.batch_size == 1

    if args.verilog and args.num_images != 1:
        sys.exit("--verilog can only be used with one image (--num_images=1)")

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST test data.
    test_images, test_labels = load_mnist_data(args.tensor_path)

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(f"{tensor_file} not found. Run tensorflow_training.py first.")

    # Create PyRTL inference hardware.
    input_bitwidth = 8
    accumulator_bitwidth = 32
    pyrtl_inference = PyRTLInference(
        quantized_model_name=tensor_file,
        input_bitwidth=input_bitwidth,
        accumulator_bitwidth=accumulator_bitwidth,
        axi=args.axi,
        initial_delay_cycles=args.initial_delay_cycles,
    )

    correct = 0
    for batch_start_index, test_batch in batched_images(
        test_images, args.start_image, args.num_images, args.batch_size
    ):
        # Print the test image.
        test_image = test_batch[0]
        print(f"PyRTL network input (#{batch_start_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_batch.dtype, "\n")

        # Run PyRTL inference on the test image.
        layer0_output, layer1_output, actual = pyrtl_inference.simulate(
            test_batch, args.verilog
        )

        # Print results.
        print(
            "PyRTL layer0 output (transposed)", layer0_output.shape, layer0_output.dtype
        )
        print(layer0_output.transpose(), "\n")

        print(
            "PyRTL layer1 output (transposed)", layer1_output.shape, layer1_output.dtype
        )
        print(layer1_output.transpose(), "\n")

        print(f"PyRTL network output (#{batch_start_index}):")
        expected = test_labels[batch_start_index]
        display_outputs(layer1_output, expected=expected, actual=actual)

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
