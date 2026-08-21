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

    `define ARRAY_S 32

    `define W_W     8
    `define A_W     4
    `define RW_W    5
    `define R_W     8

    `define RPE_MUL_W   (`RW_W + (`A_W + 1))
    `define RPE_SHIFT_W 6
    `define RPE_RES_W   (`RPE_MUL_W + `RPE_SHIFT_W)
    `define PSUM_W      (`RPE_RES_W + $clog2(`ARRAY_S))

    `define SRAM_D      512
    `define SRAM_W      45
    `define SRAM_ADDR_W $clog2(`SRAM_D)

    `define WMEM_D        8192
    `define WMEM_DW       (`RW_W * `ARRAY_S)
    `define WMEM_WMACROS  ((`WMEM_DW + `SRAM_W - 1) / `SRAM_W)
    `define WMEM_PW       (`WMEM_WMACROS * `SRAM_W)
    `define WMEM_BANKS    (`WMEM_D / `SRAM_D)
    `define WMEM_ADDR_W   $clog2(`WMEM_D)
    `define WMEM_BSEL_W   $clog2(`WMEM_BANKS)

    `define UB_D           16384
    `define UB_DW          (`A_W * `ARRAY_S)
    `define UB_WMACROS     ((`UB_DW + `SRAM_W - 1) / `SRAM_W)
    `define UB_PW          (`UB_WMACROS * `SRAM_W)
    `define UB_BANKS       (`UB_D / `SRAM_D)
    `define UB_ADDR_W      $clog2(`UB_D)
    `define UB_BSEL_W      $clog2(`UB_BANKS)

    `define WMEM_SLOT_W    $clog2(`WMEM_D / `ARRAY_S)
    `define UB_SLOT_W      $clog2(`UB_D / `ARRAY_S)
    `define ROW_IDX_W      $clog2(`ARRAY_S)

    `define W_BW           `WMEM_DW
    `define A_BW           `UB_DW
    `define R_BW           (`R_W * `ARRAY_S)
    `define PSUM_BW        (`PSUM_W * `ARRAY_S)
    `define W_RAW_BW       (`W_W * `ARRAY_S)

    `define M_MAX 1024
    `define N_MAX 4096
    `define K_MAX 8192
    `define M_W $clog2(`M_MAX + 1)
    `define N_W $clog2(`N_MAX + 1)
    `define K_W $clog2(`K_MAX + 1)

    `define ACC_W (`PSUM_W + $clog2(`K_MAX / `ARRAY_S))

    `define QMULT_W  18
    `define QSHIFT_W 6

    `define CMD_DESC_W 512
    `define CMD_OP_W   3

    `define CMD_OP_NOP       3'd0
    `define CMD_OP_LOAD_W    3'd1
    `define CMD_OP_PRELOAD_W 3'd2
    `define CMD_OP_LOAD_A    3'd3
    `define CMD_OP_GEMM      3'd4
    `define CMD_OP_STORE_C   3'd5
    `define CMD_OP_READ_UB   3'd6

    `define CMD_OP_LSB        0
    `define CMD_ACC_INIT_BIT  (`CMD_OP_LSB + `CMD_OP_W)
    `define CMD_ACC_FINAL_BIT (`CMD_ACC_INIT_BIT + 1)

    `define CMD_QMULT_LSB     8
    `define CMD_QSHIFT_LSB    (`CMD_QMULT_LSB + `QMULT_W)
    `define CMD_MT_W          $clog2(`M_MAX / `ARRAY_S)
    `define CMD_KT_W          $clog2(`K_MAX / `ARRAY_S)
    `define CMD_NT_W          $clog2(`N_MAX / `ARRAY_S)
    `define CMD_MT_LSB        (`CMD_QSHIFT_LSB + `QSHIFT_W)
    `define CMD_KT_LSB        (`CMD_MT_LSB + `CMD_MT_W)
    `define CMD_NT_LSB        (`CMD_KT_LSB + `CMD_KT_W)
    `define CMD_W_SLOT_LSB    (`CMD_NT_LSB + `CMD_NT_W)
    `define CMD_SRC_SLOT_LSB  (`CMD_W_SLOT_LSB + `WMEM_SLOT_W)
    `define CMD_DST_SLOT_LSB  (`CMD_SRC_SLOT_LSB + `UB_SLOT_W)
    `define CMD_DEFINED_W     (`CMD_DST_SLOT_LSB + `UB_SLOT_W)

`endif
