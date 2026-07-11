/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    UB_Wrapper.sv
* Project:      tpu-accelerator-lite
* Module:       UB_Wrapper
* Author:       Marco <harry2963753@gmail.com>
* Veri:         PASS
*
******************************************************************************/

module UB_Wrapper (
    input clk,
    input [`AMEM_DW-1:0] data_i,        // ub data in per row packed
    input [`AMEM_ADDR_W-1:0] addr_w,    // ub waddr
    input [`AMEM_ADDR_W-1:0] addr_r,    // ub raddr 
    input en_w,                         // ub wenable
    input en_r,                         // ub renable
    output logic [`AMEM_DW-1:0] data_o  // ub data out per row packed
);

    // memory width would be not prefectly match the data width we needed (per row) 
    localparam ZERO_PAD = `AMEM_W - `AMEM_DW;

    // memory data in and out
    logic [`AMEM_W-1:0] mdata_i;
    logic [`AMEM_W-1:0] mdata_o [0:`AMEM_BANKS-1];

    // memory bank selected
    logic [`AMEM_BSEL_W-1:0] w_bsel;
    logic [`AMEM_BSEL_W-1:0] r_bsel;
    logic [`AMEM_BSEL_W-1:0] r_bsel_q;

    // memory addr
    logic [`AMEM_MADDR_W-1:0]w_maddr;
    logic [`AMEM_MADDR_W-1:0]r_maddr;

    always_comb begin : UB_ADDR_SPLIT
        mdata_i = {{ZERO_PAD{1'b0}}, data_i};    
        w_bsel = addr_w[`AMEM_ADDR_W-1:`AMEM_MADDR_W];  // high bits: bank select
        w_maddr = addr_w[`AMEM_MADDR_W-1:0];            // low bits: intra-macro address
        r_bsel = addr_r[`AMEM_ADDR_W-1:`AMEM_MADDR_W];  // high bits: bank select
        r_maddr = addr_r[`AMEM_MADDR_W-1:0];            // low bits: intra-macro address
    end

    always_ff @(posedge clk) r_bsel_q <= r_bsel;            // reading would be needed 1 cycle delay
    always_comb data_o = mdata_o[r_bsel_q][0+:`AMEM_DW];    // select reading data

    // UB Structure : 16-bank x (16 x 88-bit) = 256 x 88-bit
    genvar b;
    generate
        for(b = 0; b < `AMEM_BANKS; b = b + 1) begin : BANK_GEN
            TS6N16ADFPCLLLVTA16X120M2FWSHOD u_Unified_Buffer(
                .CLKW(clk),
                .WEB(~(en_w & (w_bsel == b))),
                .BWEB({`AMEM_W{~(en_w & (w_bsel == b))}}),
                .AA(w_maddr),
                .D(mdata_i),
                .CLKR(clk),
                .REB(~(en_r & (r_bsel == b))),
                .AB(r_maddr),
                .Q(mdata_o[b]),
                .RCT(2'b01),
                .WCT(2'b01),
                .KP(3'b011),
                .SLP(1'd0),
                .DSLP(1'd0),
                .SD(1'd0),
                .PUDELAY());
        end
    endgenerate

endmodule
