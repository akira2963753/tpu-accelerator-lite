/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    CSA.sv
* Project:      tpu-accelerator-lite
* Module:       CSA (Compensated Systolic Array)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module CSA(
    input clk,
    input rst_n,
    input [`CW_BW-1:0] cweight,
    input cweight_valid,
    input [`A_BW-1:0] activation,
    input activation_valid,
    output logic [`ARRAY_S*`CPSUM_W-1:0] cpsum_o
);

    genvar j;
    generate
        for (j = 0; j < `ARRAY_S; j = j + 1) begin : COL_GEN
            CPE Compensated_Processing_Element(
            .clk(clk),
            .rst_n(rst_n),
            .cweight_i(cweight[`CW_W*j +: `CW_W]),
            .cweight_valid_i(cweight_valid),
            .activation_i(activation[`A_W*j +: `A_W]),
            .activation_valid_i(activation_valid),
            .cpsum_o(cpsum_o[`CPSUM_W*j +: `CPSUM_W]));
        end
    endgenerate

endmodule