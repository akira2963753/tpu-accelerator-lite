/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    Data_Setup.sv
* Project:      tpu-accelerator-lite
* Module:       Data_Setup
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module Data_Setup (
    input clk,
    input rst_n,
    input [`A_BW-1:0] activation_i,
    input [`ARRAY_S-1:0] valid_i,
    input skew_en,
    output logic [`A_BW-1:0] activation_o,
    output logic [`ARRAY_S-1:0] valid_o
);

    logic [`A_W-1:0] skew [0:`ARRAY_S-1][0:`ARRAY_S-1];
    logic valid_skew [0:`ARRAY_S-1][0:`ARRAY_S-1];

    genvar r;
    generate
        for(r = 0; r < `ARRAY_S; r++) begin : SKEW_ROW
            always_ff @(posedge clk or negedge rst_n) begin
                if(!rst_n) begin
                    for(int k = 0; k <= r; k++) begin
                        skew[r][k] <= 0;
                        valid_skew[r][k] <= 0;
                    end
                end
                else if(skew_en) begin
                    skew[r][0] <= activation_i[`A_W*r +: `A_W];
                    valid_skew[r][0] <= valid_i[r];
                    for(int k = 1; k <= r; k++) begin
                        skew[r][k] <= skew[r][k-1];
                        valid_skew[r][k] <= valid_skew[r][k-1];
                    end
                end
            end
        end
    endgenerate

    always_comb begin
        for(int r = 0; r < `ARRAY_S; r++) begin
            activation_o[`A_W*r +: `A_W] = skew[r][r];
            valid_o[r] = valid_skew[r][r];
        end
    end

endmodule
