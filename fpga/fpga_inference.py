import argparse
import pathlib
import shutil
import sys
import time

import numpy as np
import pyrtl

from pyrtlnet.cli_util import display_image, display_outputs
from pyrtlnet.constants import quantized_model_prefix
from pyrtlnet.inference_util import (
    add_common_arguments,
    batched_images,
    load_mnist_data,
    preprocess_image,
)


def main() -> None:
    """Run pyrtlnet quantized inference on a Pynq Z2 FPGA board.

    This script should be copied to the board, along with all other required assets,
    with ``make deploy``. Then this script can be run on the board with::

        $ python fpga_inference.py
    """
    # Importing pynq currently takes ~8 seconds on a Pynq Z2, so let the user know.
    start = time.time()
    print("Importing pynq... ", end="", flush=True)
    import pynq  # noqa: PLC0415

    print(f"done ({time.time() - start:.1f} seconds)")

    parser = argparse.ArgumentParser(prog="fpga_inference.py")
    add_common_arguments(parser)
    args = parser.parse_args()

    assert args.batch_size == 1

    terminal_columns = shutil.get_terminal_size((80, 24)).columns
    np.set_printoptions(linewidth=terminal_columns)

    # Load MNIST test data.
    test_images, test_labels = load_mnist_data(args.tensor_path)

    tensor_file = pathlib.Path(args.tensor_path) / f"{quantized_model_prefix}.npz"
    if not tensor_file.exists():
        sys.exit(
            f"{tensor_file} not found. Copy {quantized_model_prefix}.npz to the board "
            "first."
        )

    # Load quantization metadata. This loads the required metadata directly, instead of
    # using `SavedTensors`, to avoid a false dependency on `fxpmath`.
    tensors = np.load(tensor_file)
    input_scale = tensors.get("input.scale")
    input_zero = tensors.get("input.zero")

    for batch_start_index, test_batch in batched_images(
        test_images, args.start_image, args.num_images, args.batch_size
    ):
        # Display the test image.
        test_image = test_batch[0]
        print(f"PyRTL network input (#{batch_start_index}):")
        display_image(test_image)
        print("test_image", test_image.shape, test_image.dtype, "\n")

        # Run PyRTL FPGA inference on the test image.
        #
        # Load the pyrtlnet FPGA bitstream on the board.
        #
        # TODO: Improve the hardware design so it can run multiple images without a full
        # reset.
        print("Loading bitstream... ", end="", flush=True)
        start = time.time()
        overlay = pynq.Overlay("pyrtlnet.bit")
        print(f"done ({time.time() - start:.1f} seconds)")

        # Prepare the test image.
        flat_batch = preprocess_image(test_batch, input_scale, input_zero)
        # Convert the signed image data to raw byte values.
        for flat_image in flat_batch.transpose():
            flat_image_bytes = [
                pyrtl.infer_val_and_bitwidth(int(data), bitwidth=8, signed=True).value
                for data in flat_image
            ]

            # Find the smallest power of 2 that's larger than `len(flat_image)`.
            buffer_size = 2 ** (len(flat_image_bytes).bit_length())

            # Load the test image data in a Pynq buffer.
            buffer = pynq.allocate(shape=(buffer_size,), dtype=np.uint8)
            buffer[: len(flat_image_bytes)] = flat_image_bytes

            print("Sending image data via Pynq DMA")
            overlay.dma.sendchannel.transfer(buffer)

            # Retrieve layer1's argmax, which is stored in AXI-Lite register 0. The
            # register mapping is defined in `PyRTLInference._make_inference()`.
            print("Retrieving results\n")
            actual = overlay.pyrtlnet.read(0)
            print("pyrtlnet FPGA layer1 argmax:", actual, "\n")

            # Layer 0 outputs are in AXI-Lite registers 1-18. AXI registers are 32 bits,
            # and AXI addresses are byte addresses, so we multiply by 4.
            #
            # The values stored in each register are raw bit patterns, each representing
            # an 8-bit signed integer, so we call `val_to_signed_integer` to reinterpret
            # them as 8-bit signed integers.
            layer0_output = np.array(
                [
                    [
                        pyrtl.val_to_signed_integer(
                            overlay.pyrtlnet.read(4 * i), bitwidth=8
                        )
                    ]
                    for i in range(1, 19)
                ],
                dtype=np.int8,
            )
            print(
                "pyrtlnet FPGA layer0 output (transposed)",
                layer0_output.shape,
                layer0_output.dtype,
            )
            print(layer0_output.transpose(), "\n")

            # Layer 1 outputs are in AXI-Lite registers 19-28.
            layer1_output = np.array(
                [
                    [
                        pyrtl.val_to_signed_integer(
                            overlay.pyrtlnet.read(4 * i), bitwidth=8
                        )
                    ]
                    for i in range(19, 29)
                ],
                dtype=np.int8,
            )
            print(
                "pyrtlnet FPGA layer1 output (transposed)",
                layer1_output.shape,
                layer1_output.dtype,
            )
            print(layer1_output.transpose(), "\n")

            print(f"pyrtlnet FPGA network output (#{batch_start_index}):")
            expected = test_labels[batch_start_index]
            display_outputs(layer1_output, expected=expected, actual=actual)


if __name__ == "__main__":
    main()
