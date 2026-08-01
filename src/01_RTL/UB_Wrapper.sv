/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    UB_Wrapper.sv
* Project:      tpu-accelerator-lite
* Module:       UB_Wrapper
* Author:       Marco <harry2963753@gmail.com>
* Veri:         PASS
*
* Banked two-port activation memory. Parameterised on DEPTH so the same wrapper
* serves both users of the 16x120 macro :
*   u_UB_Wrapper   DEPTH = `AMEM_D (32)  -- activation streaming window
*   u_OBUF_Wrapper DEPTH = `OBUF_D (128) -- layer-fusion output buffer
* The ADFP N16 SRAM list only ships 16-deep macros, so DEPTH/16 macros are
* instantiated and the high address bits select the bank.
******************************************************************************/

module UB_Wrapper #(
    parameter int DEPTH  = `AMEM_D,
    parameter int ADDR_W = $clog2(DEPTH),
    parameter int BANKS  = DEPTH / `AMEM_MD,
    parameter int BSEL_W = $clog2(BANKS)
)(
    input clk,
    input [`AMEM_DW-1:0] data_i,        // data in per row packed
    input [ADDR_W-1:0] addr_w,          // write addr
    input [ADDR_W-1:0] addr_r,          // read addr
    input en_w,                         // write enable
    input en_r,                         // read enable
    output logic [`AMEM_DW-1:0] data_o  // data out per row packed
);

    // memory width would be not perfectly match the data width we needed (per row)
    localparam ZERO_PAD = `AMEM_W - `AMEM_DW;

    // memory data in and out
    logic [`AMEM_W-1:0] mdata_i;
    logic [`AMEM_W-1:0] mdata_o [0:BANKS-1];

    // memory bank selected
    logic [BSEL_W-1:0] w_bsel;
    logic [BSEL_W-1:0] r_bsel;
    logic [BSEL_W-1:0] r_bsel_q;

    // memory addr
    logic [`AMEM_MADDR_W-1:0] w_maddr;
    logic [`AMEM_MADDR_W-1:0] r_maddr;

    always_comb begin : UB_ADDR_SPLIT
        mdata_i = {{ZERO_PAD{1'b0}}, data_i};
        w_bsel  = addr_w[ADDR_W-1:`AMEM_MADDR_W];   // high bits: bank select
        w_maddr = addr_w[`AMEM_MADDR_W-1:0];        // low bits: intra-macro address
        r_bsel  = addr_r[ADDR_W-1:`AMEM_MADDR_W];   // high bits: bank select
        r_maddr = addr_r[`AMEM_MADDR_W-1:0];        // low bits: intra-macro address
    end

    always_ff @(posedge clk) r_bsel_q <= r_bsel;            // reading would be needed 1 cycle delay
    always_comb data_o = mdata_o[r_bsel_q][0+:`AMEM_DW];    // select reading data

    // Structure : BANKS x (16 x 120-bit)
    genvar b;
    generate
        for(b = 0; b < BANKS; b = b + 1) begin : BANK_GEN
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
