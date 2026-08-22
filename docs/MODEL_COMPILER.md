# TPU Model Compiler

## Scope

The first model workload executes the complete MNIST MLP on the TPU:

```text
784 -> Linear 512 -> ReLU -> Linear 256 -> ReLU -> Linear 10
```

The first layer input is loaded from the Host. Intermediate payloads remain in
UB and become the next layer's GEMM source. Every `STORE_C` is checked by RTL,
so a mismatch is localized to the first failing layer. Argmax and accuracy
reporting remain in software.

## Architecture

```text
src/00_TESTBED/compiler/
|-- pytorch_exporter.py # Generic PyTorch package exporter
|-- models/
|   `-- mnist_mlp.py    # MNIST architecture and preprocessing adapter
|-- tpu_compiler.py     # Quantization, lowering, scheduling and emission
|-- tpu_reference.py    # Bit-true inference and quantization analysis
`-- check_compiler.py   # Deterministic compiler self-check
```

The compiler consumes a portable model package and does not import PyTorch or
know MNIST-specific preprocessing. Model adapters own checkpoint loading,
architecture construction and dataset collection. The generic model format is
`tpu-model-v1`; legacy `tpu-mlp-v1` packages remain accepted.

The exported package contains:

```text
compiler/artifacts/mnist/
|-- mnist_mlp.json     # Graph, shapes and tensor names
`-- mnist_mlp.npz      # Weights, bias, calibration and MNIST samples
```

Training artifacts are not committed to Git.

## Checkpoint Export

The project checkpoint is produced by `model/MLP/train.py` and is named
`mnist_mlp_model.pth`. The compiler flow never retrains it. Run the exporter on
a machine with PyTorch, torchvision, the checkpoint and MNIST access.

Run from `src/00_TESTBED`:

```sh
python3 compiler/models/mnist_mlp.py \
    --download \
    --checkpoint ../../model/MLP/mnist_mlp_model.pth \
    --outdir compiler/artifacts/mnist
```

When the checkpoint is already under `model/MLP/`, `--checkpoint` can be
omitted:

```sh
python3 compiler/models/mnist_mlp.py \
    --download \
    --outdir compiler/artifacts/mnist
```

The exporter saves 512 calibration images, a balanced 32-image RTL batch, and
the complete 10,000-image MNIST test set. Inputs use MNIST mean `0.1307` and
standard deviation `0.3081`.

## Compiler Flow

The compiler needs Python and NumPy, but not PyTorch. Compilation and RTL test
emission are separate:

```sh
python3 compiler/check_compiler.py

python3 compiler/tpu_compiler.py compile \
    compiler/artifacts/mnist/mnist_mlp.json \
    --outdir compiler/build

python3 compiler/tpu_compiler.py emit-test \
    compiler/build/compiled_model.json \
    --inputs compiler/artifacts/mnist/mnist_mlp.json

python3 compiler/tpu_compiler.py replay pattern
```

`compile` performs model import, Type-2/A5 calibration and target lowering. It
produces `compiled_model.json` and `compiled_model.npz`, which contain no test
dataset. `emit-test` adds a selected input batch and replaces the active files
under `pattern/`. The resulting bundle checks every intermediate `STORE_C`.

The MNIST dimensions map to the hardware as follows:

| Layer | GEMM | Tiles MT x KT x NT | WMEM slots |
|---|---|---:|---:|
| fc1 | 32 x 784 x 512 | 1 x 25 x 16 | 250 maximum |
| fc2 | 32 x 512 x 256 | 1 x 16 x 8 | 128 |
| fc3 | 32 x 256 x 10 | 1 x 8 x 1 | 8 |

The first layer has 400 weight tiles, so it is split into two N blocks using
250 and 150 WMEM slots. It also exercises `K_VALID=16`. The final layer
exercises the N tail and uses `ACT_NONE`.

## Accuracy

Run the hardware-aware Python model over all exported MNIST test images:

```sh
python3 compiler/tpu_compiler.py evaluate \
    compiler/artifacts/mnist/mnist_mlp.json
```

This accuracy includes activation quantization, WPU Type-2 weight reduction,
fixed activation LSB reconstruction, 21-bit PSUM wrapping, 29-bit ACC wrapping,
bias, requantization and activation mode.

Before bit-true analysis, run the non-bit-true quantization ablation:

```sh
python3 compiler/tpu_compiler.py ablate \
    compiler/artifacts/mnist/mnist_mlp.json
```

It reports FP32, INT8 weight, Type-2 weight, ordinary INT4 activation and the
hardware A5 fixed-LSB activation accuracy. A5 has four stored payload bits and
reconstructs the MAC operand as `{payload[3:0], 1'b1}`. These stages use
floating-point MAC and bias without PSUM/ACC wrapping or integer
`QMULT/QSHIFT`.

Add `--include-bittrue` to append these progressive hardware stages:

1. Compiler activation scales with floating-point MAC and bias.
2. INT64 MAC with floating-point bias.
3. Quantized 29-bit bias.
4. Integer `QMULT/QSHIFT` and output saturation.
5. 21-bit PSUM wrapping.
6. 29-bit ACC wrapping, identical to the complete bit-true model.

The first stage whose accuracy decreases identifies the hardware rule that
needs investigation.

Compiler calibration uses percentile/MSE selection for the A5 scale. For each
layer, it derives the hardware multiplier from
`input_scale * weight_scale / (2 * output_scale)` and encodes it into the legal
18-bit `QMULT` and 6-bit `QSHIFT` fields. The ablation report also prints the
per-layer scales, encoded multiplier and mismatch rate between ideal A5
payloads and hardware requantized payloads.

The final classifier layer uses label-free rank-aware calibration. It searches
legal `QMULT/QSHIFT` pairs around the MSE solution and selects the pair that
best preserves the pre-quantization logit argmax. Ties and reconstruction MSE
are secondary criteria. Hidden layers continue to use scale-aware MSE
calibration.

## RTL Simulation

After uploading the exported JSON and NPZ package, run from `src/01_RTL`:

```sh
make -f ../00_TESTBED/makefile rtl_mnist
```

The active bundle can also be generated manually from `src/00_TESTBED`, then
simulated with `src/01_RTL/01_run`.

The Python accuracy result and RTL result have separate meanings:

- Python evaluates classification quality over 10,000 images.
- RTL verifies the selected 32-image workload bit-for-bit.
