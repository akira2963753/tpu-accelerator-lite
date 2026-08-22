# TPU Verification Flow

## Responsibility

Python owns arithmetic, tiling, memory slots, descriptor order, stream order,
golden data, and coverage contracts. `PATTERN.sv` only replays compiled files,
performs handshakes, checks stalled output stability, and compares accepted
result beats.

No `CHECKER.sv` is used.

## Directory Layout

```text
src/00_TESTBED/
|-- PATTERN.sv
|-- TESTBED.sv
|-- makefile
|-- pattern/
|   |-- test_suite.svh
|   |-- suite_summary.txt
|   `-- tNN_*.dat
|-- python/
|   |-- tpu_command.py
|   `-- tpu_gen.py
`-- directed/
    |-- gen_suite.py
    `-- check_replay.py
```

`PATTERN.sv`, `TESTBED.sv`, and the Python command/model code are shared by
directed and future model workloads. `pattern/` is the active workload staging
directory. Everything specific to synthetic regression generation is kept
under `directed/`.

## Profiles

| Profile | Cases | Purpose |
|---|---:|---|
| `regression` | 28 | Default functional regression |
| `stress` | 4 | Individual dimension limits and a larger mixed case |
| `all` | 32 | Regression followed by stress |

## Regression Cases

| ID | Case | Purpose |
|---:|---|---|
| 00 | `ub_load_read` | Basic `LOAD_A` and `READ_UB` |
| 01 | `gemm_basic` | One 32x32 GEMM tile |
| 02 | `gemm_signed` | Signed activation and both WPU paths |
| 03 | `gemm_multi_k` | Two K tiles and ACC accumulation |
| 04 | `weight_reuse` | One weight tile reused by two M tiles |
| 05 | `multi_tile_slot` | Multiple WMEM/UB slots and output tiles |
| 06 | `requant_relu` | ReLU, QMULT, QSHIFT, and saturation |
| 07 | `store_read_ub` | Store output followed by UB readback |
| 08 | `ready_stall` | Periodic STORE_C backpressure |
| 09 | `nop_command` | NOP completion without stream traffic |
| 10 | `ub_bank_boundary` | UB slots 0, 15, 16, and 511 |
| 11 | `ub_overwrite` | Same-slot overwrite and neighbor isolation |
| 12 | `wmem_bank_boundary` | WMEM slots 0, 15, 16, and 255 |
| 13 | `preload_switch_reuse` | Select W0, W1, then resident W0 again |
| 14 | `activation_reuse` | One activation tile with two weight tiles |
| 15 | `dst_slot_isolation` | Store boundary slots and read them in reverse |
| 16 | `activation_code_sweep` | All 16 activation payloads |
| 17 | `weight_code_sweep` | All 256 raw weight codes |
| 18 | `relu_mixed_sign` | Negative, zero, positive, and saturated lanes |
| 19 | `requant_boundary` | QMULT/QSHIFT zero, normal, limit, and saturation |
| 20 | `acc_init_restart` | ACC accumulation followed by ACC_INIT restart |
| 21 | `read_ub_stall` | Multi-cycle READ_UB backpressure |
| 22 | `k_valid_tail` | K_VALID=5 exact tail masking |
| 23 | `act_mode_none` | Signed ACT_NONE compared with ACT_RELU |
| 24 | `bias_init` | LOAD_BIAS, BIAS_EN, and zero initialization |
| 25 | `activation_mask` | 32x32 spatial mask combined with K_VALID=17 |
| 26 | `input_valid_gap` | LOAD_W and LOAD_A valid gaps |
| 27 | `rectangular_tiles` | M=96, K=64, N=96 tiling with stalls |

## Stress Cases

| ID | Case | Shape | Purpose |
|---:|---|---:|---|
| 00 | `m_limit` | 1024x32x32 | `M_MAX` |
| 01 | `n_limit` | 32x32x4096 | `N_MAX` |
| 02 | `k_limit` | 32x8192x32 | `K_MAX` and all 256 WMEM slots |
| 03 | `large_mixed` | 128x128x128 | Longer mixed tiling sequence |

## Generated Files

Each test uses four files under `src/00_TESTBED/pattern`:

```text
tNN_command.dat
tNN_weight.dat
tNN_activation.dat
tNN_golden.dat
```

`pattern/test_suite.svh` contains file lengths, display dimensions, names, and
stream stall modes. `pattern/suite_summary.txt` records case sizes and data
coverage. Running a generator replaces the active `tNN_*.dat` workload.

`LOAD_W` consumes 32 weight beats and `LOAD_BIAS` consumes 4 weight beats.
`LOAD_A` consumes 32 activation beats, plus 8 mask beats when `A_MASK_EN=1`.
`PATTERN.sv` derives these lengths directly from each descriptor.

## Server Flow

Run from `src/00_TESTBED`:

```sh
python3 directed/gen_suite.py --profile regression
python3 directed/check_replay.py --profile regression
```

The checker must print:

```text
PASS: all Python checks passed
```

Then run RTL simulation from `src/01_RTL`:

```sh
./01_run
```

Run one generated case with:

```sh
make -f ../00_TESTBED/makefile vcs_rtl SIM_ARGS=+TEST_ID=17
```

Stress vectors are selected with:

```sh
make -f ../00_TESTBED/makefile rtl_regress PROFILE=stress
```

The final RTL pass/fail result is determined only by Server simulation.

## MNIST MLP Flow

The model compiler keeps the directed suite intact and replaces only the active
files under `pattern/`. See `docs/MODEL_COMPILER.md` for training and export.

Run the compiler self-check and generate the complete three-layer MLP from
`src/00_TESTBED`:

```sh
python3 model/check_compiler.py
python3 model/tpu_compiler.py compile model/artifacts/mnist/mnist_mlp.json
python3 model/tpu_compiler.py replay pattern
```

Then run RTL from `src/01_RTL`:

```sh
make -f ../00_TESTBED/makefile rtl_mnist
```
