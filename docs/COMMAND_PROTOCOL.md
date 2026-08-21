# TPU Serial Command Descriptor Protocol

## Scope

The TPU exposes a direct 512-bit serial command interface. The Host issues one
command only when `cmd_ready=1` and waits for `done` before issuing the next
command. RTL does not implement a command FIFO, dependency checker, or memory
validity scoreboard.

The datapath is a 32x32 Type-2 reduced-precision systolic array. One tile holds
32 rows. WMEM exposes 256 tile slots and UB exposes 512 tile slots.

## Descriptor layout

| Bits | Field | Meaning |
|---|---|---|
| `[2:0]` | `OPCODE` | Command opcode |
| `[3]` | `ACC_INIT` | Clear ACC before a GEMM tile |
| `[4]` | `ACC_FINAL` | Final K-tile metadata |
| `[7:5]` | Reserved | Must be zero |
| `[25:8]` | `QMULT` | 18-bit requant multiplier |
| `[31:26]` | `QSHIFT` | 6-bit requant right shift |
| `[36:32]` | `MT` | M-tile metadata |
| `[44:37]` | `KT` | K-tile metadata |
| `[51:45]` | `NT` | N-tile metadata |
| `[59:52]` | `W_SLOT` | WMEM tile slot |
| `[68:60]` | `SRC_SLOT` | UB source tile slot |
| `[77:69]` | `DST_SLOT` | UB destination tile slot |
| `[511:78]` | Reserved | Must be zero |

`MT`, `KT`, `NT`, and `ACC_FINAL` describe the software schedule. The current
RTL directly consumes the slot fields and `ACC_INIT`.

## Opcodes

| Value | Opcode | Effect |
|---:|---|---|
| `0` | `NOP` | Complete without a datapath action |
| `1` | `LOAD_W` | Accept 32 raw 256-bit weight rows into `W_SLOT` |
| `2` | `PRELOAD_W` | Load the 32x32 RSA from `W_SLOT` |
| `3` | `LOAD_A` | Accept 32 128-bit activation rows into `SRC_SLOT` |
| `4` | `GEMM` | Read `SRC_SLOT` and accumulate one tile |
| `5` | `STORE_C` | Write 32 result payload rows to `DST_SLOT` and stream them to Host |
| `6` | `READ_UB` | Read 32 rows from `SRC_SLOT` and stream them to Host |

`PRELOAD_W` remains separate from `LOAD_W`. WMEM can retain multiple weight
tiles while the RSA holds one active tile.

## Memory organization

WMEM is 8192x160-bit and uses a 13-bit logical row address:

```text
address[12:5] = 8-bit tile slot
address[4:0]  = row inside the 32-row tile
```

UB is 16384x128-bit and uses a 14-bit logical row address:

```text
address[13:5] = 9-bit tile slot
address[4:0]  = row inside the 32-row tile
```

WMEM uses 64 `TS1N16ADFPCLLLVTA512X45M4SWSHOD` macros:

```text
16 depth banks x 4 width macros = 8192x180 physical bits
```

UB uses 96 instances of the same macro:

```text
32 depth banks x 3 width macros = 16384x135 physical bits
```

Only the lower 160 WMEM bits and lower 128 UB bits are data. Padding bits are
written as zero. Both memories are single-port because commands execute
serially.

## Data formats

The WPU converts each raw 8-bit weight to a 5-bit code:

```text
MSR-4     : {1'b0, weight[4:1]}
Non-MSR-4 : {1'b1, weight[7:4]}
```

Each RPE reconstructs fixed-LSB operands and performs one signed 5x5 multiply:

```text
weight_5b     = {weight_payload, 1'b1}
activation_5b = {activation_payload, 1'b1}
contribution  = shift_flag ? product << 6 : product << 3
```

Activation input and UB payloads are 4-bit per lane. Output processing applies
ReLU, requantization, and saturation before writing UB. Data returned to the
Host is expanded to 8-bit per lane:

```text
r_data_lane = {payload[3:0], 4'b0000}
```

`r_data` is therefore 256-bit. Both `STORE_C` and `READ_UB` use this external
format, while UB remains 128-bit per row.

## Limits

```text
M_MAX = 1024
N_MAX = 4096
K_MAX = 8192
```

All hardware tiles are 32x32. Software pads dimensions that are not multiples
of 32 and owns the outer tile loop, memory lifetime, and command ordering.
