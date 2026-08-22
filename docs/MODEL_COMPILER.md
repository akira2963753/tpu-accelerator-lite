# MNIST MLP Compiler

## Scope

The first model workload executes the complete MNIST MLP on the TPU:

```text
784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
```

The first layer input is loaded from the Host. Intermediate payloads remain in
UB and become the next layer's GEMM source. Every `STORE_C` is checked by RTL,
so a mismatch is localized to the first failing layer. Argmax and accuracy
reporting remain in software.

## Files

```text
src/00_TESTBED/model/
|-- mnist_mlp.py       # PyTorch training and portable model export
|-- tpu_compiler.py    # Quantization, tiling, scheduling and bundle replay
`-- check_compiler.py  # Deterministic 784-256-128-10 self-check
```

The exported package contains:

```text
model/artifacts/mnist/
|-- mnist_mlp.json     # Graph, shapes and tensor names
|-- mnist_mlp.npz      # Weights, bias, calibration and MNIST samples
`-- mnist_mlp.pth      # Optional PyTorch checkpoint
```

Training artifacts are not committed to Git.

## Model Export

Run this step on a machine with PyTorch, torchvision and MNIST access. Run from
`src/00_TESTBED`:

```sh
python3 model/mnist_mlp.py \
    --download \
    --epochs 5 \
    --outdir model/artifacts/mnist
```

To export an existing checkpoint without retraining:

```sh
python3 model/mnist_mlp.py \
    --download \
    --checkpoint model/artifacts/mnist/mnist_mlp.pth \
    --outdir model/artifacts/mnist
```

The exporter saves 512 calibration images, a balanced 32-image RTL batch, and
the complete 10,000-image MNIST test set. Inputs use MNIST mean `0.1307` and
standard deviation `0.3081`.

## Compiler Flow

The compiler needs Python and NumPy, but not PyTorch:

```sh
python3 model/check_compiler.py

python3 model/tpu_compiler.py compile \
    model/artifacts/mnist/mnist_mlp.json

python3 model/tpu_compiler.py replay pattern
```

`compile` replaces the active files under `pattern/`. The bundle contains one
test with all three layers, including intermediate `STORE_C` results.

The MNIST dimensions map to the hardware as follows:

| Layer | GEMM | Tiles MT x KT x NT | WMEM slots |
|---|---|---:|---:|
| fc1 | 32 x 784 x 256 | 1 x 25 x 8 | 200 |
| fc2 | 32 x 256 x 128 | 1 x 8 x 4 | 32 |
| fc3 | 32 x 128 x 10 | 1 x 4 x 1 | 4 |

The first layer exercises `K_VALID=16`. The final layer exercises the N tail
and uses `ACT_NONE`.

## Accuracy

Run the hardware-aware Python model over all exported MNIST test images:

```sh
python3 model/tpu_compiler.py evaluate \
    model/artifacts/mnist/mnist_mlp.json
```

This accuracy includes activation quantization, WPU Type-2 weight reduction,
fixed activation LSB reconstruction, 21-bit PSUM wrapping, 29-bit ACC wrapping,
bias, requantization and activation mode.

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
