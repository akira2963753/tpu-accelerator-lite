/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    CHIP_TOP.sv
* Project:      tpu-accelerator-lite
* Module:       CHIP_TOP
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/
`include "define.vh"

module CHIP_TOP (
    input logic clk,
    input logic rst_n,

    input logic [`HOST_DATA_W-1:0] host_data_i,
    input logic [`HOST_TYPE_W-1:0] host_type,
    input logic host_valid,
    output logic host_ready,

    output logic [`HOST_DATA_W-1:0] host_data_o,
    output logic host_out_valid,
    input logic host_out_ready,

    output logic busy,
    output logic done
);

    logic cmd_valid, cmd_ready;
    logic [`CMD_DESC_W-1:0] cmd_desc;
    logic w_valid, w_ready;
    logic [`W_RAW_BW-1:0] w_data;
    logic a_valid, a_ready;
    logic [`A_BW-1:0] a_data;
    logic r_valid, r_ready;
    logic [`R_BW-1:0] r_data;
    logic core_busy, core_done;

    TPU_IO_BRIDGE u_TPU_IO_BRIDGE (
        .clk(clk),
        .rst_n(rst_n),
        .host_data_i(host_data_i),
        .host_type(host_type),
        .host_valid(host_valid),
        .host_ready(host_ready),
        .host_data_o(host_data_o),
        .host_out_valid(host_out_valid),
        .host_out_ready(host_out_ready),
        .core_cmd_valid(cmd_valid),
        .core_cmd_ready(cmd_ready),
        .core_cmd_desc(cmd_desc),
        .core_w_valid(w_valid),
        .core_w_ready(w_ready),
        .core_w_data(w_data),
        .core_a_valid(a_valid),
        .core_a_ready(a_ready),
        .core_a_data(a_data),
        .core_r_valid(r_valid),
        .core_r_ready(r_ready),
        .core_r_data(r_data),
        .core_busy(core_busy),
        .core_done(core_done),
        .host_busy(busy),
        .host_done(done)
    );

    TPU u_TPU (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_desc(cmd_desc),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .w_data(w_data),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .a_data(a_data),
        .r_valid(r_valid),
        .r_ready(r_ready),
        .r_data(r_data),
        .busy(core_busy),
        .done(core_done)
    );

endmodule
