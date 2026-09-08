/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    CHIP.v
* Project:      tpu-accelerator-lite
* Module:       CHIP
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/
module CHIP (
    clk,
    rst_n,
    host_data_i,
    host_type,
    host_valid,
    host_ready,
    host_data_o,
    host_out_valid,
    host_out_ready,
    busy,
    done
);

    input clk;
    input rst_n;
    input [7:0] host_data_i;
    input [1:0] host_type;
    input host_valid;
    output host_ready;
    output [7:0] host_data_o;
    output host_out_valid;
    input host_out_ready;
    output busy;
    output done;

    wire ioc_clk, ioc_rst_n;
    wire [7:0] ioc_host_data_i;
    wire [1:0] ioc_host_type;
    wire ioc_host_valid, ioc_host_ready;
    wire [7:0] ioc_host_data_o;
    wire ioc_host_out_valid, ioc_host_out_ready;
    wire ioc_busy, ioc_done;

    //=============================================================
    //                         Core Instance
    //=============================================================
    CHIP_TOP u_CHIP_TOP (
        .clk(ioc_clk),
        .rst_n(ioc_rst_n),
        .host_data_i(ioc_host_data_i),
        .host_type(ioc_host_type),
        .host_valid(ioc_host_valid),
        .host_ready(ioc_host_ready),
        .host_data_o(ioc_host_data_o),
        .host_out_valid(ioc_host_out_valid),
        .host_out_ready(ioc_host_out_ready),
        .busy(ioc_busy),
        .done(ioc_done)
    );

    //=============================================================
    //                   Left Input Pads, H Type
    //=============================================================
    PDCDG_H ipad_clk (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(clk),
        .C(ioc_clk)
    );

    PDCDG_H ipad_rst_n (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(rst_n),
        .C(ioc_rst_n)
    );

    PDCDG_H ipad_host_type_0 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_type[0]),
        .C(ioc_host_type[0])
    );

    PDCDG_H ipad_host_type_1 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_type[1]),
        .C(ioc_host_type[1])
    );

    PDCDG_H ipad_host_valid (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_valid),
        .C(ioc_host_valid)
    );

    PDCDG_H ipad_host_out_ready (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_out_ready),
        .C(ioc_host_out_ready)
    );

    PDCDG_H ipad_host_data_i_0 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[0]),
        .C(ioc_host_data_i[0])
    );

    //=============================================================
    //                   Top Input Pads, V Type
    //=============================================================
    PDCDG_V ipad_host_data_i_1 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[1]),
        .C(ioc_host_data_i[1])
    );

    PDCDG_V ipad_host_data_i_2 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[2]),
        .C(ioc_host_data_i[2])
    );

    PDCDG_V ipad_host_data_i_3 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[3]),
        .C(ioc_host_data_i[3])
    );

    PDCDG_V ipad_host_data_i_4 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[4]),
        .C(ioc_host_data_i[4])
    );

    PDCDG_V ipad_host_data_i_5 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[5]),
        .C(ioc_host_data_i[5])
    );

    PDCDG_V ipad_host_data_i_6 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[6]),
        .C(ioc_host_data_i[6])
    );

    PDCDG_V ipad_host_data_i_7 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[7]),
        .C(ioc_host_data_i[7])
    );

    //=============================================================
    //                  Right Output Pads, H Type
    //=============================================================
    PDCDG_H opad_host_data_o_0 (
        .I(ioc_host_data_o[0]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[0]),
        .C()
    );

    PDCDG_H opad_host_data_o_1 (
        .I(ioc_host_data_o[1]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[1]),
        .C()
    );

    PDCDG_H opad_host_data_o_2 (
        .I(ioc_host_data_o[2]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[2]),
        .C()
    );

    PDCDG_H opad_host_data_o_3 (
        .I(ioc_host_data_o[3]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[3]),
        .C()
    );

    PDCDG_H opad_host_data_o_4 (
        .I(ioc_host_data_o[4]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[4]),
        .C()
    );

    PDCDG_H opad_host_data_o_5 (
        .I(ioc_host_data_o[5]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[5]),
        .C()
    );

    //=============================================================
    //                 Bottom Output Pads, V Type
    //=============================================================
    PDCDG_V opad_host_data_o_6 (
        .I(ioc_host_data_o[6]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[6]),
        .C()
    );

    PDCDG_V opad_host_data_o_7 (
        .I(ioc_host_data_o[7]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[7]),
        .C()
    );

    PDCDG_V opad_host_ready (
        .I(ioc_host_ready),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_ready),
        .C()
    );

    PDCDG_V opad_host_out_valid (
        .I(ioc_host_out_valid),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_out_valid),
        .C()
    );

    PDCDG_V opad_busy (
        .I(ioc_busy),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(busy),
        .C()
    );

    PDCDG_V opad_done (
        .I(ioc_done),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(done),
        .C()
    );

endmodule
