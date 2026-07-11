/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    RSA.sv
* Project:      tpu-accelerator-lite
* Module:       RSA (Reduced Systolic Array)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module RSA(
    input clk,
    input rst_n,
    input weight_valid,
    input [`W_BW-1:0] weight,
    input [`A_BW-1:0] activation,
    output logic [`PSUM_BW-1:0] psum
);

    logic [`RW_W-1:0] weight_net [0:`ARRAY_S][0:`ARRAY_S-1];
    logic [`A_W-1:0] activation_net [0:`ARRAY_S-1][0:`ARRAY_S];
    logic weight_valid_net [0:`ARRAY_S][0:`ARRAY_S-1];
    logic [`PSUM_W-1:0] psum_net [0:`ARRAY_S][0:`ARRAY_S-1];

    genvar i, j;
    generate
        for(i = 0; i < `ARRAY_S; i++) begin : PREPROCESS_BLOCK
            always_comb begin
                weight_net[0][i] = weight[`RW_W*i +: `RW_W];
                weight_valid_net[0][i] = weight_valid;
                activation_net[i][0] = activation[`A_W*i +: `A_W];
                psum_net[0][i] = 0;
                psum[`PSUM_W*i +: `PSUM_W] = psum_net[`ARRAY_S][i];
            end
        end
    endgenerate

    generate
        for (i = 0; i < `ARRAY_S; i = i + 1) begin : ROW_GEN
            for (j = 0; j < `ARRAY_S ; j = j + 1) begin : COL_GEN
                RPE Reduced_Processing_Element(
                .clk(clk),
                .rst_n(rst_n),
                .weight_i(weight_net[i][j]),
                .activation_i(activation_net[i][j]),
                .psum_i(psum_net[i][j]),
                .weight_valid_i(weight_valid_net[i][j]),
                .weight_o(weight_net[i+1][j]),
                .weight_valid_o(weight_valid_net[i+1][j]),
                .activation_o(activation_net[i][j+1]),
                .psum_o(psum_net[i+1][j]));
            end
        end
    endgenerate

endmodule