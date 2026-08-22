# TPU Model Compiler Specification

## Status

- Version: `1.0-approved`
- Scope: Conv2D and Linear/FC inference
- Target: 32x32 Type-2 reduced-precision TPU
- Purpose: Freeze the software/hardware contract before RTL and compiler work

This document is the approved software/hardware contract. Features marked
**New** are additions relative to the previous command protocol.

## Design Goals

The TPU accelerates only the matrix multiplication portion of Conv2D and
Linear layers. Software owns model parsing, quantization, tensor layout,
tiling, memory lifetime, command scheduling, and unsupported graph operators.

The compiler shall:

1. Lower Conv2D and Linear layers to tiled GEMM operations.
2. Generate legal serial command descriptors and stream data.
3. Schedule workloads larger than WMEM or UB without requiring all tensors to
   be resident at once.
4. Produce bit-exact golden output for RTL replay.
5. Keep the existing directed regression flow intact.

## Target Hardware

| Item | Value |
|---|---:|
| Systolic array | 32x32 |
| Activation payload | Signed 4-bit |
| Reconstructed activation | `{payload[3:0], 1'b1}` |
| Raw weight input | Signed 8-bit |
| Reduced weight | 5-bit WPU code |
| Accumulator | Signed 29-bit |
| WMEM | 8192x160-bit, 256 tile slots |
| UB | 16384x128-bit, 512 tile slots |
| Maximum M per GEMM problem | 1024 |
| Maximum K per GEMM problem | 8192 |
| Maximum N per GEMM problem | 4096 |

WMEM and UB are managed as 32-row tile slots. Commands execute serially. The
Host issues a new command only after the previous command completes.

## Required Hardware Extensions

The model compiler depends on four additions to the current hardware.

### K_VALID

**New.** `K_VALID` specifies how many reduction lanes in the current GEMM tile
are valid.

- Legal range for `GEMM`: 1 to 32.
- A full K tile is encoded as 32, not 0.
- Non-GEMM commands set the field to 0.
- Only reduction indices `0` through `K_VALID-1` may contribute.
- Invalid lanes contribute an exact arithmetic zero inside the datapath.

This cannot be replaced by input padding. The Type-2 fixed LSB makes every
encoded activation and weight an odd, non-zero 5-bit value.

### Activation Validity Mask

**New.** `A_MASK_EN` allows Conv2D zero-padding to contribute an exact zero.
`K_VALID` alone is insufficient because padding validity can differ for every
M row and K position in an im2col tile.

The hardware keeps one active 32x32 activation mask in registers. It is not
stored for every UB slot.

- `A_MASK_EN=0`: all 1024 activation elements are valid and `LOAD_A` consumes
  the normal 32 payload beats.
- `A_MASK_EN=1`: `LOAD_A` consumes 32 payload beats followed by 8 mask beats.
- Mask bit 1 enables the corresponding multiplication; bit 0 forces its
  contribution to an exact arithmetic zero before accumulation.
- The effective GEMM enable is the activation mask bit AND `K_VALID`.
- The active mask persists until the next `LOAD_A` command.

Mask packing is row-major:

```text
element_index     = m_lane * 32 + k_lane
mask_beat         = element_index / 128
mask_bit          = element_index % 128
a_data[mask_bit]  = valid[m_lane][k_lane]
```

The first payload beat after accepting `LOAD_A` remains activation row 0; mask
beats begin only after activation row 31. A GEMM using a masked activation tile
must execute before another `LOAD_A` replaces the active mask. The compiler
schedules all desired N-tile reuse while that mask is active. RTL does not need
to check that the source slot and active mask correspond.

This active-mask design avoids adding a 16384x32-bit mask memory beside UB.

### Programmable Activation Mode

**New.** Output processing supports:

| Value | Mode | Result range |
|---:|---|---:|
| `0` | `ACT_NONE` | Signed 4-bit, -8 to 7 |
| `1` | `ACT_RELU` | Unsigned non-negative payload, 0 to 7 |
| `2` | Reserved | - |
| `3` | Reserved | - |

The mode belongs to `STORE_C`, together with `QMULT` and `QSHIFT`.

For `ACT_NONE`:

```text
scaled  = (acc * QMULT) >>> QSHIFT
payload = saturate(scaled, -8, 7)
```

For `ACT_RELU`:

```text
scaled  = (max(acc, 0) * QMULT) >>> QSHIFT
payload = saturate(scaled, 0, 7)
```

`QMULT` remains restricted to 0 through 131071. `QSHIFT` is 0 through 63.
The UB stores the resulting 4-bit payload. Host readback remains:

```text
r_data_lane = {payload[3:0], 4'b0000}
```

For `ACT_NONE`, software interprets `r_data_lane[7:4]` as signed two's
complement. A result stored in UB can be used directly as the next layer's
activation payload.

### Bias Load and Initialization

**New.** Opcode 7 is assigned to `LOAD_BIAS`.

- Bias storage is one active vector of 32 signed 29-bit values.
- Reset initializes the vector to zero.
- The vector persists until another `LOAD_BIAS` command.
- One vector is reusable across all M tiles of the same N tile.
- No separate bias memory or bias slot field is added.

`LOAD_BIAS` reuses the existing 256-bit weight stream and consumes exactly four
accepted beats. Each beat contains eight signed 32-bit containers:

```text
lane          = beat_index * 8 + container_index
w_data field  = w_data[32*container_index +: 32]
```

The compiler sign-extends every signed 29-bit bias into its 32-bit container.
The RTL stores bits `[28:0]`; bits `[31:29]` must match the sign bit.

Bias is already quantized into the accumulator domain. On a `GEMM` command:

- `ACC_INIT=1, BIAS_EN=0`: initialize the 32x32 ACC tile to zero.
- `ACC_INIT=1, BIAS_EN=1`: initialize every M row with the active 32-lane bias.
- `ACC_INIT=0`: continue accumulation and require `BIAS_EN=0`.

The Host guarantees legal command order. RTL does not need a bias-valid
scoreboard or dependency checker.

## Command Descriptor

The existing descriptor fields remain unchanged. New fields occupy previously
reserved bits.

| Bits | Field | Command | Meaning |
|---|---|---|---|
| `[2:0]` | `OPCODE` | All | Command opcode |
| `[3]` | `ACC_INIT` | GEMM | Initialize ACC before this K tile |
| `[4]` | `ACC_FINAL` | GEMM | Final K-tile metadata |
| `[7:5]` | Reserved | All | Must be zero |
| `[25:8]` | `QMULT` | STORE_C | Non-negative requant multiplier |
| `[31:26]` | `QSHIFT` | STORE_C | Requant arithmetic right shift |
| `[36:32]` | `MT` | Scheduled op | M-tile metadata |
| `[44:37]` | `KT` | Scheduled op | K-tile metadata |
| `[51:45]` | `NT` | Scheduled op | N-tile metadata |
| `[59:52]` | `W_SLOT` | Weight op | WMEM tile slot |
| `[68:60]` | `SRC_SLOT` | UB read op | UB source tile slot |
| `[77:69]` | `DST_SLOT` | UB write op | UB destination tile slot |
| `[83:78]` | `K_VALID` | GEMM | Valid reduction lanes, 1 to 32 |
| `[84]` | `BIAS_EN` | GEMM | Initialize ACC from active bias |
| `[86:85]` | `ACT_MODE` | STORE_C | Output activation function |
| `[87]` | `A_MASK_EN` | LOAD_A | Receive and activate a 32x32 validity mask |
| `[511:88]` | Reserved | All | Must be zero |

After this extension, `CMD_DEFINED_W` is 88.

## Opcodes

| Value | Opcode | Stream traffic | Effect |
|---:|---|---|---|
| `0` | `NOP` | None | Complete without datapath work |
| `1` | `LOAD_W` | 32 weight beats | Reduce and store one WMEM tile |
| `2` | `PRELOAD_W` | None | Load one WMEM tile into the RSA |
| `3` | `LOAD_A` | 32 or 40 activation beats | Store one UB tile and set its active mask |
| `4` | `GEMM` | None | Accumulate one 32x32 tile product |
| `5` | `STORE_C` | 32 result beats | Requantize, store to UB, and return |
| `6` | `READ_UB` | 32 result beats | Read one UB tile and return |
| `7` | `LOAD_BIAS` | 4 weight beats | Replace the active bias vector |

## Supported Model Semantics

### Linear

For a PyTorch-style Linear layer with weight shape `[N, K]`:

```text
input  -> flatten leading dimensions to [M, K]
weight -> transpose to [K, N]
output -> [M, N]
```

Bias is optional. The compiler chooses `ACT_NONE` or `ACT_RELU` from the
legalized graph.

### Conv2D

For NCHW input and OIHW weight:

```text
M = batch * output_height * output_width
K = (input_channels / groups) * kernel_height * kernel_width
N = output_channels / groups
```

Each group is lowered to one logical GEMM. Stride, padding, dilation, and
group indexing are performed by software while generating activation tiles.
The compiler generates im2col data tile-by-tile and does not materialize the
full im2col matrix.

M and N tail positions are stored but ignored by software. K tails use
`K_VALID`. Spatial zero-padding uses the activation validity mask. Encoded data
values are never used as arithmetic zero padding.

### Graph Operations

The first compiler version supports:

- Conv2D
- Linear/FC
- ReLU fused into the preceding `STORE_C`
- BatchNorm folded into Conv/Linear weight and bias when legal
- Flatten, reshape, and layout transforms in software

The Host executes operations that are not mapped to the TPU, including:

- Residual Add
- Pooling
- Softmax
- Arbitrary element-wise operations
- Unsupported activation functions

## Quantization Contract

The reference path starts from FP32 parameters and calibration samples.

1. Quantize weights to symmetric signed INT8.
2. Pass each INT8 weight through the exact WPU Type-2 reduction model.
3. Quantize activations to signed 4-bit payloads.
4. Include fixed-LSB reconstruction in the bit-exact arithmetic model.
5. Quantize bias into the signed 29-bit accumulator domain.
6. Derive non-negative `QMULT` and `QSHIFT` for output requantization.

Calibration uses a representative dataset, not only the maximum value of a
single tensor. Version 1 uses one `QMULT` and `QSHIFT` pair per output tile or
layer. Per-channel requantization is not supported by the current descriptor.

The compiler must report saturation counts for weights, activations, bias, and
outputs. Overflow of the signed 29-bit accumulator is a compile-time error.

## Compiler Architecture

The compiler exposes one public driver, `tpu_compiler.py`. Internal stages may
be functions or classes in the same implementation; the command-line flow must
remain integrated.

```text
PyTorch model and calibration data
              |
              v
       Frontend Model IR
              |
              v
   Legalize and fold operators
              |
              v
   Hardware-aware quantization
              |
              v
      Conv/Linear GEMM IR
              |
              v
  Tile and blocking scheduler
              |
              v
 Virtual slot allocation/liveness
              |
              v
 Descriptor and stream backend
              |
              v
 Bit-exact trace replay and bundle
```

### Internal Representations

The implementation shall keep these concepts separate even if they reside in
one Python file:

- `TargetSpec`: array size, widths, memory capacities, and descriptor layout.
- `TensorIR`: shape, layout, quantization parameters, and lifetime.
- `LayerIR`: legalized Conv2D or Linear operation.
- `GemmIR`: logical M, K, N problem and fused bias/activation.
- `TileIR`: virtual activation, validity mask, weight, bias, and output tiles.
- `CommandIR`: typed command before descriptor packing.
- `WorkloadBundle`: descriptors, streams, golden data, and metadata.

Descriptor encoding remains centralized in the shared
`src/00_TESTBED/python/tpu_command.py` module.

## Tiling and Scheduling

Logical tile counts are:

```text
MT = ceil(M / 32)
KT = ceil(K / 32)
NT = ceil(N / 32)
```

A model layer may exceed the per-problem M or N limits. The compiler splits it
into independent M chunks of at most 1024 rows and N chunks of at most 4096
columns, then restores the logical output order. K must not exceed 8192 for one
Conv group or Linear layer in version 1.

The scheduler uses virtual tiles first, then maps live tiles to physical slots.
It must not assume that all unique model weights or activations fit at once.

For an M block of `MB` tiles and an N block of `NB` tiles:

```text
resident weight tiles = NB * KT <= 256
live UB tiles          = MB * KT + live output tiles <= 512
```

These equations are initial feasibility checks. The final allocator uses exact
lifetimes and may stream smaller K blocks when `KT` itself prevents residency.

For each output tile, all K tiles execute consecutively while ACC holds the
partial sum:

1. Load or select the N tile's bias.
2. Set `ACC_INIT=1` and optional `BIAS_EN=1` on the first GEMM.
3. Set `ACC_INIT=0` on later GEMMs.
4. Set `K_VALID` for every GEMM.
5. Issue `STORE_C` with output requantization and activation mode.

The cost model considers WMEM reloads, UB reloads, weight reuse across M tiles,
activation reuse across N tiles, and output lifetime. Correctness is mandatory;
the cost model only chooses among legal schedules.

## Workload Bundle

Generated model workloads are separate from the existing directed suite:

```text
src/00_TESTBED/
|-- pattern/                   # Active workload staging directory
|   |-- manifest.json          # Model workload only
|   |-- test_suite.svh
|   |-- suite_summary.txt
|   `-- tNN_*.dat
|-- directed/                  # Existing synthetic regression tools
|-- python/                    # Shared command and arithmetic model
`-- model/
    |-- tpu_compiler.py        # Integrated compiler CLI
    `-- check_compiler.py      # Compiler self-check
```

`manifest.json` records source model information, input shapes, quantization,
layer shapes, schedule decisions, slot usage, command counts, stream sizes,
saturation statistics, and expected tensor checksums.

`PATTERN.sv` remains a generic file replay engine. Directed generators and the
model compiler both replace the active bundle under `pattern/`. The include and
data paths never change, and arithmetic or scheduling must not move into
SystemVerilog.

## Compiler CLI

The initial public interface is:

```text
python3 model/tpu_compiler.py compile <model> [options]
python3 model/tpu_compiler.py replay <bundle>
python3 model/check_compiler.py
```

The `compile` command imports, calibrates, quantizes, schedules, emits, and
replays the workload before reporting success. A bundle is valid only when its
independent bit-exact replay passes.

## Verification Contract

The model flow adds coverage; it does not replace directed verification.

The acceptance sequence is:

1. Existing directed regression passes after descriptor and RTL changes.
2. Python unit tests cover descriptor fields, K tails, spatial padding masks,
   bias, and both activation modes.
3. Independent naive Linear and Conv2D references match compiler trace replay.
4. Representative individual layers match RTL output bit-for-bit.
5. A multi-layer MLP with UB chaining matches the hardware-aware Python model.
6. ResNet/AlexNet Conv and FC layers compile without exceeding physical slots.
7. End-to-end model accuracy is measured in Python with the same quantization
   semantics used by RTL.

RTL simulation compares packed output payloads exactly. FP32 model accuracy is
a separate quality metric and does not replace bit-exact checking.

## Explicit Non-Goals

Version 1 does not provide:

- CPU-facing RISC ISA or AXI integration
- Multiple commands in flight
- Runtime command dependency checking
- On-chip residual Add or pooling
- Full-model weight residency
- Per-channel output requantization
- Training or backpropagation
- Dynamic tensor shapes inside one compiled bundle

## Implementation Order

1. Freeze this specification.
2. Add `K_VALID`, `A_MASK_EN`, `ACT_MODE`, `LOAD_BIAS`, and `BIAS_EN` to RTL and
   the directed reference model.
3. Restore and extend directed RTL regression.
4. Implement descriptor/IR/scheduler self-checks.
5. Implement Linear compilation and a chained MLP workload.
6. Implement tiled Conv2D lowering and real Conv layer workloads.
7. Compile ResNet18, AlexNet, and representative MLP layers.
8. Measure command counts, memory traffic, cycles, saturation, and accuracy.

The descriptor layout and the four required hardware extensions are frozen.
Any incompatible change requires a new specification revision.
