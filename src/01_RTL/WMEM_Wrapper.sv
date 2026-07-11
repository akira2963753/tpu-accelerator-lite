/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    WMEM_Wrapper.sv
* Project:      tpu-accelerator-lite
* Module:       WMEM_Wrapper
* Author:       Marco <harry2963753@gmail.com>
* Veri:         PASS
*
******************************************************************************/

module WMEM_Wrapper (
    input clk,
    input [`WMEM_ADDR_W-1:0] addr_w,    // wmem waddr
    input [`WMEM_DW-1:0] data_i,        // wmem data in (per row packed)
    input en_c,                         // chip enable
    input en_w,                         // write enable
    output logic [`WMEM_DW-1:0] data_o  // wmem data out (per row packed)
);

    // memory width would be not prefectly match the data width we needed (per row) 
    localparam ZERO_PAD = `WMEM_W - `WMEM_DW;

    // memory data in and out
    logic [`WMEM_W-1:0] mdata_i;
    logic [`WMEM_W-1:0] mdata_o [0:`WMEM_BANKS-1];

    // memory bank selected
    logic [`WMEM_BSEL_W-1:0] bsel;
    logic [`WMEM_BSEL_W-1:0] bsel_q;

    // memory addr
    logic [`WMEM_MADDR_W-1:0] maddr;

    always_comb begin : WMEM_ADDR_SPLIT
        mdata_i = {{ZERO_PAD{1'b0}}, data_i};    
        bsel = addr_w[`WMEM_ADDR_W-1:`WMEM_MADDR_W];    // high bits: bank select
        maddr = addr_w[`WMEM_MADDR_W-1:0];              // low bits: intra-macro address
    end

    always_ff @(posedge clk) bsel_q <= bsel;            // reading would be needed 1 cycle delay
    always_comb data_o = mdata_o[bsel_q][0+:`WMEM_DW];  // select reading data

    // WMem Structure : 4-bank x (16 x 88-bit) = 64 x 88-bit
    genvar b;
    generate
        for(b = 0; b < `WMEM_BANKS; b = b + 1) begin : BANK_GEN
            TS1N16ADFPCLLLVTA16X88M2SWSHOD u_Weight_Mem(
                .CLK(clk),
                .CEB(~(en_c & (bsel == b))),
                .WEB(~(en_w & (bsel == b))),
                .A(maddr),
                .D(mdata_i),
                .Q(mdata_o[b]),
                .BWEB({`WMEM_W{~(en_w & (bsel == b))}}),
                .SLP(1'd0),
                .DSLP(1'd0),
                .SD(1'd0),
                .PUDELAY(),
                .RTSEL(2'b01),
                .WTSEL(2'b01));
        end
    endgenerate

endmodule
