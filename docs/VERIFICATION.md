# TPU Verification Flow

## Responsibility

Python owns the arithmetic model, tiling, memory slots, descriptor sequence,
input stream order, and expected output order. `PATTERN.sv` only replays the
compiled files, performs handshakes, and compares accepted output beats.

No `CHECKER.sv` is used in the primary verification stage.

## Generated Files

Each test uses four files under `src/00_TESTBED/pattern`:

```text
tNN_command.dat
tNN_weight.dat
tNN_activation.dat
tNN_golden.dat
```

The input and golden files follow command consumption order. `PATTERN.sv`
therefore uses sequential pointers and does not calculate tile addresses.

`test_suite.svh` contains only file lengths, display dimensions, names, and the
output-stall option.

## Primary Cases

| Case | Purpose |
|---|---|
| `ub_load_read` | `LOAD_A` followed by `READ_UB` |
| `gemm_basic` | One 32x32 GEMM tile |
| `gemm_signed` | Signed activation and MSR/Non-MSR weight paths |
| `gemm_multi_k` | Two K tiles and ACC accumulation |
| `weight_reuse` | One resident weight tile reused by two M tiles |
| `multi_tile_slot` | Multiple WMEM/UB slots and output tiles |
| `requant_relu` | ReLU, QMULT, QSHIFT, and 4-bit output |
| `store_read_ub` | `STORE_C` output followed by UB readback |
| `ready_stall` | Deterministic `r_ready` backpressure |

## Server Flow

Run from `src/00_TESTBED`:

```sh
python3 python/gen_suite.py
python3 python/check_replay.py
```

The second command must print:

```text
PASS: all Python checks passed
```

Then run RTL simulation from `src/01_RTL`:

```sh
./01_run
```

`01_run` invokes VCS through `src/00_TESTBED/makefile`. The final RTL pass/fail
result is determined only by the Server simulation.
