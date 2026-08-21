# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 16x16 systolic-array INT8 TPU-v1-style inference accelerator that exploits the
**MSR-4** (Most Significant Runs of length 4) property of trained weights: ~99% of
INT8 weights have `w[7:4]` all-0 or all-1, so they can be carried as 5 bits
(`{shift_bit, w[4:1]}`) instead of 8. The rare non-MSR-4 weights are recovered by a
separate 1x16 **compensation array**, so accuracy is not sacrificed.

Target: TSMC 16nm (N16ADFP), Design Compiler + IC Compiler II, VCS/Verdi.
EDA tools and SRAM macros live on a remote CAD server (`/usr/cad/ADFP/...`); nothing
in `src/` simulates or synthesizes locally on Windows.

## Commands

Simulation runs **from `src/01_RTL/`** — `file.f` uses paths relative to that
directory, and `PATTERN.sv` reads its `.dat` via `../00_TESTBED/pattern/`.

```bash
cd src/01_RTL
make -f ../00_TESTBED/makefile gen_suite      # generate data + software descriptor traces
make -f ../00_TESTBED/makefile check_replay   # validate trace/data mapping
make -f ../00_TESTBED/makefile vcs_rtl        # whole serial-command suite, no waves
make -f ../00_TESTBED/makefile rtl_regress    # gen_suite -> check_replay -> VCS regression
make -f ../00_TESTBED/makefile vcs_rtl_dump   # same + FSDB (opt-in: suite is ~3.2M cycles)
make -f ../00_TESTBED/makefile vcs_gate       # needs ../02_SYN/Netlist/TPU_syn.{v,sdf}
make -f ../00_TESTBED/makefile clean
```

Synthesis runs from `src/02_SYN/` (its own `file.f`, so `Netlist/` lands where
`TESTBED.sv`'s `$sdf_annotate` looks for it):

```bash
cd src/02_SYN
dc_shell -f syn16.tcl | tee syn.log           # DESIGN=TPU, CYCLE=2ns, ss0p72v125c
```

Generating the verification suite (host machine, standard-library Python only):

```bash
cd src/00_TESTBED
python3 python/gen_suite.py                   # default tier (serial-command cycle estimate in suite_summary.txt)
python3 python/gen_suite.py --big             # + the 1024^3 case
python3 python/check_replay.py                # verify the .dat replay contract (fast)
```

`gen_suite.py` is the **single source of truth** for the test list; `test_suite.svh`
and `suite_summary.txt` are generated — never hand-edit them. To add a test, append
a `T(...)` row to `TEST_TABLE` and rerun. `pattern/*.dat` is gitignored.

The current Host-facing interface and legal command schedules are documented in
[`docs/COMMAND_PROTOCOL.md`](docs/COMMAND_PROTOCOL.md).

Fast standalone control check (~1 s, run this before the full suite):

```bash
cd src/10_VERI/TSC/00_TESTBED && make -f ../../makefile vcs_rtl
```

Other per-block envs live in `src/10_VERI/<BLOCK>/` (TSC, RSA, WPU, UB, WMEM), each
with `00_TESTBED` / `02_SYN` / `03_GATE` and its own `file.f`. Note RSA and WPU still
carry 90nm synthesis scripts (`syn90.tcl`, `CYCLE=15`) while the top is 16nm/2ns.

## Architecture

### Dataflow and module map

Weight-stationary. `TPU.sv` is a thin structural top; `TSC.sv` holds *all* control.

```
w_data(8b x16) -> WPU -> {rweight 5b x16 -> WMEM_Wrapper -> RSA (16x16 RPE)}
                      -> {cweight 4b x16 -> CMEM        -> CSA (1x16 CPE)}
a_data(7b x16) -> UB_Wrapper  ─┐
OBUF (fusion)  ────────────────┴ act_sel mux -> Data_Setup -> {skew -> RSA ; comp row -> CSA}
                                          RSA.psum ┐
                                          CSA.cpsum┴-> ACC (16x16 x 28b) -> OP -> r_data
                                                                              └-> OBUF
```

- **WPU** splits each raw 8-bit weight in one combinational lane per column, then
  registers the result (1-cycle latency — `TSC` delays the WMEM write strobe by one
  cycle via `wmem_wr_q` to match).
- **Data_Setup** holds the triangular skew array *and* the per-column compensation
  bookkeeping (`comp_row[j]`, `comp_valid[j]`).
- **ACC** is a flop-based 16x16 register file, read combinationally so `r_data` is
  valid in the same cycle as `r_valid`.

### Memories — why they are shaped the way they are

The ADFP N16 SRAM list only ships **16-word-deep** macros, so any capacity is paid
for with a full periphery tax per macro. That single fact drives the memory design:

| | depth | macros | role |
|---|---|---|---|
| `WMEM_Wrapper` | 32 | 2 | one 16-row weight tile; bank 1 reserved for a future load/preload ping-pong |
| `UB_Wrapper` (`u_UB_Wrapper`) | 32 | 2 | **activation streaming window**, 2 k-tiles |
| `UB_Wrapper` (`u_OBUF_Wrapper`) | 128 | 8 | layer-fusion output buffer |

`UB_Wrapper` is parameterised on `DEPTH`; both instances use the same 16x120 macro.

**The UB is not a full-K activation cache.** One k-tile (16 rows) is streamed in per
`(mt,nt,kt)` and consumed by the following `GEMM`. That is what decouples UB depth
from K — otherwise `K_MAX=1024` would need 128 macros. A K=16 command schedule
keeps each preloaded `(nt,0)` weight tile resident across all `mt` tiles; K>16 still
reloads each K tile because ACC holds one output tile at a time.

### The MAC decomposition (RPE + CPE are one arithmetic unit)

This is the least obvious part of the design and must be changed as a pair:

- MSR-4 weight (`shift_bit=0`): payload is `w[4:1]`, LSB forced to 1. `RMAC` computes
  `act_ext * (2*payload + 1)` where `act_ext = {a,1'b1} = 2a+1`. Negation uses a real
  two's complement (`~p + 1`) because `weight_sign_add = payload_sign && !shift_bit`.
- Non-MSR-4 weight (`shift_bit=1`): payload is `w[7:4]`, contributing `payload<<4`,
  and `CMEM`/`CPE` supply `{w[7], w[3:1], 1'b1}` as a **signed 5-bit** term.
  `RMAC` deliberately negates with **one's complement** (`~p`, i.e. magnitude `|p|-1`)
  here; the missing `-16` is exactly cancelled by the sign bit of the signed
  compensation term. Fix one side without the other and every negative non-MSR-4
  weight is off by 16*activation.

### Serial descriptor controller (`TSC.sv`)

The current controller accepts one 512-bit command descriptor only while idle:
`IDLE -> {LOAD_W | PRELOAD_W | LOAD_A | GEMM | STORE_C} -> DONE -> IDLE`.
`cmd_ready` is high only in `IDLE`; `done` is a one-cycle completion pulse. The
Host, not the RTL, guarantees resource order and waits for `cmd_ready` before
every command. See [`docs/COMMAND_PROTOCOL.md`](docs/COMMAND_PROTOCOL.md) for the
field map and legal schedules.

`LOAD_W` and `PRELOAD_W` are deliberately separate. Once preloaded, a 16x16
weight tile remains resident in RSA/CSA, so K=16 workloads can reuse it across
multiple M tiles. `ACC_INIT` clears ACC at the beginning of each output tile;
`STORE_C` is separate and applies requantization only after the full K reduction.

### Historical whole-job controller (superseded)

`IDLE -> LOADING -> PRELOAD_W -> CAL -> (LOADING | OUT) -> (LOADING | DONE)`

Loop order is `mt -> nt -> kt`. Per k-tile: 17 LOADING + 17 PRELOAD_W + 52 CAL = 86
cycles for 16 activation vectors, i.e. **~18% MAC utilisation** — the headline
optimisation left on the table is a shadow weight register per RPE plus a skewed
weight-switch, which would hide LOADING/PRELOAD and most of the drain.

Key constants: `RESULT_LAT = ARRAY_S+1` (column-0 psum latency, empirically pinned),
`CAL_DRAIN = 2*ARRAY_S+4` (must cover `RESULT_LAT` plus the 15-cycle column de-skew
tail). The `en_chain`/`row_chain` shift registers in `TSC` *are* the psum de-skew:
lane `j` is lane `j-1` delayed one cycle, driving `ACC`'s per-column write enable and
row address.

### Layer fusion (`fuse_in` / `fuse_out`)

`FUSE_IN` / `FUSE_OUT` are descriptor fields latched for the relevant command.
`fuse_out` also writes each result row to
`OBUF[nt*16 + m]`; `fuse_in` reads activation from `OBUF[kt*16 + m]` instead of the
host — the *same address formula*, which is why the producer's writes replay exactly
as the consumer's reads. A fused job asks for zero activation beats.

### Job dimension limits

`M <= 1024`, `N <= 1024`, `K <= 4096`, all multiples of 16. M and N cost only
descriptor port + counter width; **K is the expensive one** because a job's whole
K-reduction lands in one accumulator before requant, so `ACC_W = PSUM_W +
$clog2(K_MAX/ARRAY_S) = 28` and `max|C| = 4096*127*127 = 66M < 2^27`.

K is now split across **GEMM descriptors within one output tile**: `ACC_INIT`
marks the first K tile and preserves ACC through later K tiles until `STORE_C`.
Partial sums still cannot cross a `STORE_C` boundary, because `OP` applies ReLU +
requant and `r_data` is only 7-bit.

### Hard constraints — assertions exist in PATTERN, not in the RTL

`PATTERN.sv`'s SVA block (`+define+SVA`) checks these; the RTL itself does not:

- The Host derives valid `mt`/`kt`/`nt` tile indices from dimensions that are
  multiples of 16 and within `M_MAX`/`K_MAX`/`N_MAX`; TSC does not validate them.
- Commands obey the serial resource sequence documented in
  `docs/COMMAND_PROTOCOL.md`; TSC intentionally contains no validity/dependency
  scoreboard.
- `fuse_out` needs `M <= 16` (the write address ignores `mt`) and `N <= OBUF_D`.
- `fuse_in` needs `M <= 16` and `K <= OBUF_D`.
- **At most one non-MSR-4 weight per column per 16-row tile.** `CMEM` and
  `Data_Setup.comp_row[]` hold exactly one entry per column; a second overwrites the
  first. `gen_suite.py`'s `comp_ovf_16` case breaks this on purpose and is
  reverse-judged (a mismatch is the pass condition, via `TEST_XMISS`).
- K padded to a multiple of 16 is **not** safe: the expected-value LSB makes a padded
  `k` contribute `act_ext(0) * (0|1) = 1`, not 0, and odd*odd can never be 0. Real
  layers with `K % 16 != 0` need a per-row valid mask in `RPE` — not implemented.

### The .dat replay contract

Python compiles the Host command schedule into `tNN_command.dat`; `PATTERN.sv`
only replays that trace and its associated unique data. Get an index wrong and
every case fails mysteriously:

| file | rows | PATTERN index |
|---|---|---|
| `tNN_command.dat` | `TEST_CMD_COUNT[t]` | sequential command `c` |
| `tNN_weight.dat` | `NT*KT*16` | `(nt*KT + kt)*16 + r` |
| `tNN_activation.dat` | `MT*KT*16` | `(mt*KT + kt)*16 + m` |
| `tNN_golden.dat` | `MT*NT*16` | `(mt*NT + nt)*16 + m` |

`python/check_replay.py` validates the software descriptor compiler and both data
maps — run it after touching `tpu_command.py`, `gen_case` row ordering, or PATTERN's
stream transactors.

### define.vh

Not `include`d by any RTL file — it is compiled **first** in `file.f` and relies on
VCS's single compilation unit. Any new file added to `file.f` must go after it, and
any tool that compiles files separately (Verilator, most linters) will fail until the
files get explicit `` `include "define.vh" `` (the header guard already exists).

## Conventions

- SystemVerilog `.sv`, one banner comment block per file (`Copyright / File Name /
  Project / Module / Author`), 4-space indent, `always_ff`/`always_comb` never bare
  `always`.
- Testbench stimulus is driven on **negedge** so the same PATTERN works for gate sim.
- Mismatches in `PATTERN.sv` are `$display`'d (first 8 only), not `$fatal`'d, so the
  whole suite runs to completion; only the watchdog uses `$fatal`.
- `model/*.py` (MLP, LeNet, ResNet-18, AlexNet) are the PyTorch PTQ experiments that
  produced the MSR-4 statistics in the README; they are independent of the RTL flow.
  Their real GEMM dims (MLP K=784, LeNet K=400, ResNet K up to 4608, AlexNet K/N up
  to 4096) are what the dimension limits above are sized against.
- `README.md` still describes the **old** pre-`42266cf` design (256x256, `.v` files,
  directories that no longer exist, dead links). It has not been updated for this
  architecture.
