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
    input [31:0] host_data_i;
    input [1:0] host_type;
    input host_valid;
    output host_ready;
    output [31:0] host_data_o;
    output host_out_valid;
    input host_out_ready;
    output busy;
    output done;

    wire ioc_clk, ioc_rst_n;
    wire [31:0] ioc_host_data_i;
    wire [1:0] ioc_host_type;
    wire ioc_host_valid, ioc_host_ready;
    wire [31:0] ioc_host_data_o;
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
    //                  Left Input Pads, V Type
    //=============================================================
    PDCDG_V ipad_clk (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(clk),
        .C(ioc_clk)
    );

    PDCDG_V ipad_rst_n (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(rst_n),
        .C(ioc_rst_n)
    );

    PDCDG_V ipad_host_type_0 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_type[0]),
        .C(ioc_host_type[0])
    );

    PDCDG_V ipad_host_type_1 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_type[1]),
        .C(ioc_host_type[1])
    );

    PDCDG_V ipad_host_valid (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_valid),
        .C(ioc_host_valid)
    );

    PDCDG_V ipad_host_out_ready (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_out_ready),
        .C(ioc_host_out_ready)
    );

    PDCDG_V ipad_host_data_i_0 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[0]),
        .C(ioc_host_data_i[0])
    );

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

    PDCDG_V ipad_host_data_i_8 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[8]),
        .C(ioc_host_data_i[8])
    );

    PDCDG_V ipad_host_data_i_9 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[9]),
        .C(ioc_host_data_i[9])
    );

    PDCDG_V ipad_host_data_i_10 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[10]),
        .C(ioc_host_data_i[10])
    );

    PDCDG_V ipad_host_data_i_11 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[11]),
        .C(ioc_host_data_i[11])
    );

    //=============================================================
    //                  Top Input Pads, H Type
    //=============================================================
    PDCDG_H ipad_host_data_i_12 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[12]),
        .C(ioc_host_data_i[12])
    );

    PDCDG_H ipad_host_data_i_13 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[13]),
        .C(ioc_host_data_i[13])
    );

    PDCDG_H ipad_host_data_i_14 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[14]),
        .C(ioc_host_data_i[14])
    );

    PDCDG_H ipad_host_data_i_15 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[15]),
        .C(ioc_host_data_i[15])
    );

    PDCDG_H ipad_host_data_i_16 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[16]),
        .C(ioc_host_data_i[16])
    );

    PDCDG_H ipad_host_data_i_17 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[17]),
        .C(ioc_host_data_i[17])
    );

    PDCDG_H ipad_host_data_i_18 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[18]),
        .C(ioc_host_data_i[18])
    );

    PDCDG_H ipad_host_data_i_19 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[19]),
        .C(ioc_host_data_i[19])
    );

    PDCDG_H ipad_host_data_i_20 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[20]),
        .C(ioc_host_data_i[20])
    );

    PDCDG_H ipad_host_data_i_21 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[21]),
        .C(ioc_host_data_i[21])
    );

    PDCDG_H ipad_host_data_i_22 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[22]),
        .C(ioc_host_data_i[22])
    );

    PDCDG_H ipad_host_data_i_23 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[23]),
        .C(ioc_host_data_i[23])
    );

    PDCDG_H ipad_host_data_i_24 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[24]),
        .C(ioc_host_data_i[24])
    );

    PDCDG_H ipad_host_data_i_25 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[25]),
        .C(ioc_host_data_i[25])
    );

    PDCDG_H ipad_host_data_i_26 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[26]),
        .C(ioc_host_data_i[26])
    );

    PDCDG_H ipad_host_data_i_27 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[27]),
        .C(ioc_host_data_i[27])
    );

    PDCDG_H ipad_host_data_i_28 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[28]),
        .C(ioc_host_data_i[28])
    );

    PDCDG_H ipad_host_data_i_29 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[29]),
        .C(ioc_host_data_i[29])
    );

    PDCDG_H ipad_host_data_i_30 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[30]),
        .C(ioc_host_data_i[30])
    );

    PDCDG_H ipad_host_data_i_31 (
        .I(1'b0),
        .IE(1'b1),
        .OEN(1'b1),
        .PAD(host_data_i[31]),
        .C(ioc_host_data_i[31])
    );

    //=============================================================
    //                 Right Output Pads, V Type
    //=============================================================
    PDCDG_V opad_host_data_o_0 (
        .I(ioc_host_data_o[0]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[0]),
        .C()
    );

    PDCDG_V opad_host_data_o_1 (
        .I(ioc_host_data_o[1]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[1]),
        .C()
    );

    PDCDG_V opad_host_data_o_2 (
        .I(ioc_host_data_o[2]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[2]),
        .C()
    );

    PDCDG_V opad_host_data_o_3 (
        .I(ioc_host_data_o[3]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[3]),
        .C()
    );

    PDCDG_V opad_host_data_o_4 (
        .I(ioc_host_data_o[4]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[4]),
        .C()
    );

    PDCDG_V opad_host_data_o_5 (
        .I(ioc_host_data_o[5]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[5]),
        .C()
    );

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

    PDCDG_V opad_host_data_o_8 (
        .I(ioc_host_data_o[8]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[8]),
        .C()
    );

    PDCDG_V opad_host_data_o_9 (
        .I(ioc_host_data_o[9]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[9]),
        .C()
    );

    PDCDG_V opad_host_data_o_10 (
        .I(ioc_host_data_o[10]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[10]),
        .C()
    );

    PDCDG_V opad_host_data_o_11 (
        .I(ioc_host_data_o[11]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[11]),
        .C()
    );

    PDCDG_V opad_host_data_o_12 (
        .I(ioc_host_data_o[12]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[12]),
        .C()
    );

    PDCDG_V opad_host_data_o_13 (
        .I(ioc_host_data_o[13]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[13]),
        .C()
    );

    PDCDG_V opad_host_data_o_14 (
        .I(ioc_host_data_o[14]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[14]),
        .C()
    );

    PDCDG_V opad_host_data_o_15 (
        .I(ioc_host_data_o[15]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[15]),
        .C()
    );

    PDCDG_V opad_host_data_o_16 (
        .I(ioc_host_data_o[16]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[16]),
        .C()
    );

    PDCDG_V opad_host_data_o_17 (
        .I(ioc_host_data_o[17]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[17]),
        .C()
    );

    //=============================================================
    //                Bottom Output Pads, H Type
    //=============================================================
    PDCDG_H opad_host_data_o_18 (
        .I(ioc_host_data_o[18]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[18]),
        .C()
    );

    PDCDG_H opad_host_data_o_19 (
        .I(ioc_host_data_o[19]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[19]),
        .C()
    );

    PDCDG_H opad_host_data_o_20 (
        .I(ioc_host_data_o[20]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[20]),
        .C()
    );

    PDCDG_H opad_host_data_o_21 (
        .I(ioc_host_data_o[21]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[21]),
        .C()
    );

    PDCDG_H opad_host_data_o_22 (
        .I(ioc_host_data_o[22]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[22]),
        .C()
    );

    PDCDG_H opad_host_data_o_23 (
        .I(ioc_host_data_o[23]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[23]),
        .C()
    );

    PDCDG_H opad_host_data_o_24 (
        .I(ioc_host_data_o[24]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[24]),
        .C()
    );

    PDCDG_H opad_host_data_o_25 (
        .I(ioc_host_data_o[25]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[25]),
        .C()
    );

    PDCDG_H opad_host_data_o_26 (
        .I(ioc_host_data_o[26]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[26]),
        .C()
    );

    PDCDG_H opad_host_data_o_27 (
        .I(ioc_host_data_o[27]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[27]),
        .C()
    );

    PDCDG_H opad_host_data_o_28 (
        .I(ioc_host_data_o[28]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[28]),
        .C()
    );

    PDCDG_H opad_host_data_o_29 (
        .I(ioc_host_data_o[29]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[29]),
        .C()
    );

    PDCDG_H opad_host_data_o_30 (
        .I(ioc_host_data_o[30]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[30]),
        .C()
    );

    PDCDG_H opad_host_data_o_31 (
        .I(ioc_host_data_o[31]),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_data_o[31]),
        .C()
    );

    PDCDG_H opad_host_ready (
        .I(ioc_host_ready),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_ready),
        .C()
    );

    PDCDG_H opad_host_out_valid (
        .I(ioc_host_out_valid),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(host_out_valid),
        .C()
    );

    PDCDG_H opad_busy (
        .I(ioc_busy),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(busy),
        .C()
    );

    PDCDG_H opad_done (
        .I(ioc_done),
        .IE(1'b0),
        .OEN(1'b0),
        .PAD(done),
        .C()
    );

endmodule
