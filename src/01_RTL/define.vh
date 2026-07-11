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
    `define WMEM_W  88                      // WMEM macro physical width
    `define WMEM_MD 16                      // WMEM macro physical depth (one bank)
    `define WMEM_D  64                      // WMEM logical (banked) depth
    `define WMEM_BANKS (`WMEM_D / `WMEM_MD) // number of WMEM banks = 4
    `define WMEM_S  (`WMEM_D * `WMEM_W)
    `define WMEM_DW (`RW_W * `ARRAY_S)      // WMEM data bus (one reduced-weight row) = 80
    `define AMEM_W  120                     // UB macro physical width
    `define AMEM_MD 16                      // UB macro physical depth (one bank)
    `define AMEM_D  256                     // UB logical (banked) depth
    `define AMEM_BANKS (`AMEM_D / `AMEM_MD) // number of UB banks = 16
    `define AMEM_S  (`AMEM_D * `AMEM_W)
    `define AMEM_DW (`A_W * `ARRAY_S)       // UB data bus (one activation row) = 112
    `define WMEM_ADDR_W  $clog2(`WMEM_D)    // full WMEM address = 6
    `define AMEM_ADDR_W  $clog2(`AMEM_D)    // full UB address = 8
    `define WMEM_MADDR_W $clog2(`WMEM_MD)   // intra-macro address = 4
    `define AMEM_MADDR_W $clog2(`AMEM_MD)   // intra-macro address = 4
    `define WMEM_BSEL_W  $clog2(`WMEM_BANKS)// WMEM bank-select = 2
    `define AMEM_BSEL_W  $clog2(`AMEM_BANKS)// UB bank-select = 4
    `define ROW_IDX_W    $clog2(`ARRAY_S)   // array / compensation row index = 4
    `define W_BW `WMEM_DW               // weight bus width       
    `define A_BW `AMEM_DW               // activation bus width
    `define PSUM_BW `PSUM_W * `ARRAY_S  // Partial sum bus width
    `define W_RAW_BW `W_W * `ARRAY_S    // raw weight bus width
    `define CW_BW `CW_W * `ARRAY_S      // compensation bus width

    `define M_MAX 256
    `define N_MAX 256
    `define K_MAX 256
    `define M_W $clog2(`M_MAX + 1)
    `define N_W $clog2(`N_MAX + 1)
    `define K_W $clog2(`K_MAX + 1)

    `define ACC_W (`PSUM_W + $clog2(`K_MAX / `ARRAY_S)) // accumulator width : psum + cross-k_tile headroom = 24

    `define QMULT_W  18  // per-tensor requant multiplier M0 (signed fixed-point)
    `define QSHIFT_W 5   // per-tensor requant right-shift n

`endif

