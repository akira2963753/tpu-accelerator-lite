/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    CMEM.sv
* Project:      tpu-accelerator-lite
* Module:       CMEM (Compensation Memory)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module CMEM (
    input clk,
    input rst_n,
    input [`ARRAY_S-1:0] cout_valid,        // per-column Non-MSR-4 write mask
    input [`CW_BW-1:0] cweight,             // 16 x 4-bit compensation weight (packed)
    output logic [`CW_BW-1:0] cweight_o     // 16 x 4-bit stored compensation weight (packed)
);

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            cweight_o <= 0;
        end
        else begin
            for(int i = 0; i < `ARRAY_S; i = i + 1) begin
                if(cout_valid[i]) begin
                    cweight_o[`CW_W*i +: `CW_W] <= cweight[`CW_W*i +: `CW_W];
                end
            end
        end
    end

endmodule
