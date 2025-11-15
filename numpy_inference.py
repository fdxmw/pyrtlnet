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
from pyrtlnet.numpy_inference import NumPyInference


def main() -> None:
    parser = argparse.ArgumentParser(prog="numpy_inference.py")
    add_common_arguments(parser)
    args = parser.parse_args()

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST test data.
    test_images, test_labels = load_mnist_data(args.tensor_path)

    # Validate arguments.
    if args.num_images == 1:
        args.verbose = True

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(f"{tensor_file} not found. Run tensorflow_training.py first.")

    # Collect weights, biases, and quantization metadata.
    numpy_inference = NumPyInference(quantized_model_name=tensor_file)

    correct = 0
    for batch_num, (batch_start_index, test_batch) in enumerate(
        batched_images(test_images, args.start_image, args.num_images, args.batch_size)
    ):
        layer0_outputs, layer1_outputs, actuals = numpy_inference.run(test_batch)

        layer0_outputs = layer0_outputs.transpose()
        layer1_outputs = layer1_outputs.transpose()

        for batch_index in range(len(test_batch)):
            test_image = test_batch[batch_index]
            expected = test_labels[batch_start_index + batch_index]

            if batch_index > 0:
                print()

            print(
                f"NumPy network input (#{batch_start_index + batch_index}, ",
                f"batch {batch_num}, batch_index {batch_index})",
            )

            if args.verbose:
                display_image(test_image)
                print("test_image", test_image.shape, test_image.dtype, "\n")

                print(
                    "NumPy layer0 output (transposed)",
                    layer0_outputs[batch_index].shape,
                    layer0_outputs[batch_index].dtype,
                )
                print(layer0_outputs[batch_index], "\n")
                print(
                    "NumPy layer1 output (transposed)",
                    layer1_outputs[batch_index].shape,
                    layer1_outputs[batch_index].dtype,
                )
                print(layer1_outputs[batch_index], "\n")
                print(f"NumPy network output (#{batch_start_index + batch_index}):")
                display_outputs(
                    layer1_outputs[batch_index],
                    expected=expected,
                    actual=actuals[batch_index],
                )
            else:
                print(f"Expected: {expected} | Actual: {actuals[batch_index]}")

            if actuals[batch_index] == expected:
                correct += 1

        print()
    if args.num_images > 1:
        print(
            f"{correct}/{args.num_images} correct predictions, "
            f"{100.0 * correct / args.num_images:.0f}% accuracy"
        )


if __name__ == "__main__":
    main()
