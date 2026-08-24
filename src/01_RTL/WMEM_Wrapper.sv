/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    WMEM_Wrapper.sv
* Project:      tpu-accelerator-lite
* Module:       WMEM_Wrapper
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module WMEM_Wrapper (
    input clk,
    input [`WMEM_ADDR_W-1:0] addr,
    input [`WMEM_DW-1:0] data_i,
    input en,
    input we,
    output logic [`WMEM_DW-1:0] data_o
);

    localparam PAD_W = `WMEM_PW - `WMEM_DW;

    logic [`WMEM_PW-1:0] macro_data_i;
    logic [`WMEM_PW-1:0] macro_data_o [0:`WMEM_BANKS-1];
    logic [`WMEM_BSEL_W-1:0] bsel;
    logic [`WMEM_BSEL_W-1:0] bsel_q;
    logic [`SRAM_ADDR_W-1:0] maddr;

    always_comb begin : WMEM_ADDR_SPLIT
        macro_data_i = {{PAD_W{1'b0}}, data_i};
        bsel = addr[`WMEM_ADDR_W-1:`SRAM_ADDR_W];
        maddr = addr[`SRAM_ADDR_W-1:0];
    end

    always_ff @(posedge clk) begin : BANK_SELECT
        if(en && !we) bsel_q <= bsel;
    end

    always_comb begin : DATA_OUTPUT
        data_o = macro_data_o[bsel_q][0 +: `WMEM_DW];
    end

    generate
        for(genvar b = 0; b < `WMEM_BANKS; b++) begin : BANK_GEN
            for(genvar m = 0; m < `WMEM_WMACROS; m++) begin : WIDTH_GEN
                TS1N16ADFPCLLLVTA512X45M4SWSHOD u_Weight_Memory (
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
                    .WTSEL(2'b01)
                );
            end
        end
    endgenerate

endmodule
