/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    UB_Wrapper.sv
* Project:      tpu-accelerator-lite
* Module:       UB_Wrapper
* Author:       Marco <harry2963753@gmail.com>
*
* Banked single-port memory wrapper.
******************************************************************************/

module UB_Wrapper (
    input clk,
    input [`UB_ADDR_W-1:0] addr,
    input [`UB_DW-1:0] data_i,
    input en,
    input we,
    output logic [`UB_DW-1:0] data_o
);

    localparam PAD_W = `UB_PW - `UB_DW;

    logic [`UB_PW-1:0] macro_data_i;
    logic [`UB_PW-1:0] macro_data_o [0:`UB_BANKS-1];
    logic [`UB_BSEL_W-1:0] bsel;
    logic [`UB_BSEL_W-1:0] bsel_q;
    logic [`SRAM_ADDR_W-1:0] maddr;

    always_comb begin : UB_ADDR_SPLIT
        macro_data_i = {{PAD_W{1'b0}}, data_i};
        bsel = addr[`UB_ADDR_W-1:`SRAM_ADDR_W];
        maddr = addr[`SRAM_ADDR_W-1:0];
    end

    always_ff @(posedge clk) begin
        if(en && !we) bsel_q <= bsel;
    end

    always_comb data_o = macro_data_o[bsel_q][0 +: `UB_DW];

    genvar b, m;
    generate
        for(b = 0; b < `UB_BANKS; b = b + 1) begin : BANK_GEN
            for(m = 0; m < `UB_WMACROS; m = m + 1) begin : WIDTH_GEN
                TS1N16ADFPCLLLVTA512X45M4SWSHOD u_Unified_Buffer(
                    .CLK(clk),
                    .CEB(~(en && (bsel == b))),
                    .WEB(~(en && we && (bsel == b))),
                    .A(maddr),
                    .D(macro_data_i[`SRAM_W*m +: `SRAM_W]),
                    .Q(macro_data_o[b][`SRAM_W*m +: `SRAM_W]),
                    .BWEB({`SRAM_W{~(en && we && (bsel == b))}}),
                    .SLP(1'b0),
                    .DSLP(1'b0),
                    .SD(1'b0),
                    .PUDELAY(),
                    .RTSEL(2'b01),
                    .WTSEL(2'b01));
            end
        end
    endgenerate

endmodule
