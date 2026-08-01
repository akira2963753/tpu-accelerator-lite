/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    define.vh
* Project:      tpu-accelerator-lite
* Module:       define
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

`ifndef DEFINE_H
`define DEFINE_H

    `define ARRAY_S 16  // Systolic Array dimension
    `define W_W     8   // Raw weight input width
    `define A_W     7   // Activation input/output width
    `define RW_W    5   // MSR-4 reduced weight width
    `define CW_W    4   // Compensation weight width
    `define PSUM_W  (`W_W + (`A_W + 1) + $clog2(`ARRAY_S)) // Partial sum width
    `define CPSUM_W ((`CW_W + 1) + (`A_W + 1)) // Compensation partial sum width

    // ---- WMEM : one 16-row weight tile, double-buffer slot reserved ----
    // Only rows 0..15 are used today (bank 0) ; bank 1 is reserved for the
    // load/preload ping-pong that hides LOADING behind CAL.
    `define WMEM_W  88                      // WMEM macro physical width
    `define WMEM_MD 16                      // WMEM macro physical depth (one bank)
    `define WMEM_D  32                      // WMEM logical (banked) depth
    `define WMEM_BANKS (`WMEM_D / `WMEM_MD) // number of WMEM banks = 2
    `define WMEM_S  (`WMEM_D * `WMEM_W)
    `define WMEM_DW (`RW_W * `ARRAY_S)      // WMEM data bus (one reduced-weight row) = 80

    // ---- UB : activation *streaming window*, NOT a full-K cache ----
    // One k-tile (16 rows) is streamed in per (mt,nt,kt) and consumed by the
    // following CAL. Two halves alternate on kt[0] so a later prefetch can
    // write ahead without disturbing the half being read. Because UB depth no
    // longer scales with K, K_MAX is bounded only by ACC_W / port width.
    `define AMEM_W  120                     // UB macro physical width
    `define AMEM_MD 16                      // UB macro physical depth (one bank)
    `define AMEM_D  32                      // UB logical depth = 2 k-tiles
    `define AMEM_BANKS (`AMEM_D / `AMEM_MD) // number of UB banks = 2
    `define AMEM_S  (`AMEM_D * `AMEM_W)
    `define AMEM_DW (`A_W * `ARRAY_S)       // UB data bus (one activation row) = 112

    // ---- OBUF : on-chip layer-fusion output buffer ----
    // OUT writes result row m of column block nt to OBUF[nt*16 + m]. The next
    // layer reads it as activation row (kt', m) at OBUF[kt'*16 + m] -- the same
    // formula, so a fused job needs no host activation traffic at all.
    // Valid only for : fuse_out -> M <= ARRAY_S, N <= OBUF_D
    //                  fuse_in  -> M <= ARRAY_S, K <= OBUF_D
    `define OBUF_D  128                     // OBUF logical depth (8 banks)
    `define OBUF_BANKS (`OBUF_D / `AMEM_MD) // = 8
    `define OBUF_TILES (`OBUF_D / `ARRAY_S) // = 8 addressable 16-row tiles

    `define WMEM_ADDR_W  $clog2(`WMEM_D)    // full WMEM address = 5
    `define AMEM_ADDR_W  $clog2(`AMEM_D)    // full UB address = 5
    `define OBUF_ADDR_W  $clog2(`OBUF_D)    // full OBUF address = 7
    `define WMEM_MADDR_W $clog2(`WMEM_MD)   // intra-macro address = 4
    `define AMEM_MADDR_W $clog2(`AMEM_MD)   // intra-macro address = 4
    `define WMEM_BSEL_W  $clog2(`WMEM_BANKS)// WMEM bank-select = 1
    `define AMEM_BSEL_W  $clog2(`AMEM_BANKS)// UB bank-select = 1
    `define ROW_IDX_W    $clog2(`ARRAY_S)   // array / compensation row index = 4
    `define W_BW `WMEM_DW               // weight bus width
    `define A_BW `AMEM_DW               // activation bus width
    `define PSUM_BW `PSUM_W * `ARRAY_S  // Partial sum bus width
    `define W_RAW_BW `W_W * `ARRAY_S    // raw weight bus width
    `define CW_BW `CW_W * `ARRAY_S      // compensation bus width

    // ---- job dimension limits ----
    // M / N cost only descriptor port + counter width (host tiles them freely).
    // K is the expensive one : it is capped by ACC_W, since a job's whole
    // K-reduction lands in one accumulator before requant.
    `define M_MAX 1024
    `define N_MAX 1024
    `define K_MAX 4096
    `define M_W $clog2(`M_MAX + 1)
    `define N_W $clog2(`N_MAX + 1)
    `define K_W $clog2(`K_MAX + 1)

    // accumulator width : psum + cross-k_tile headroom = 28
    // max |C| = K_MAX * 127 * 127 = 66,064,384 < 2^27 (signed 28-bit) OK
    `define ACC_W (`PSUM_W + $clog2(`K_MAX / `ARRAY_S))

    `define QMULT_W  18  // per-tensor requant multiplier M0 (signed fixed-point)
    // QSHIFT_W = 6 : with M0 normalised to [2^16,2^17) and acc up to 2^26, the
    // shift needed to land back in [0,63] reaches ~37 -- a 5-bit field (max 31)
    // silently under-shifts and saturates every output.
    `define QSHIFT_W 6   // per-tensor requant right-shift n

`endif

