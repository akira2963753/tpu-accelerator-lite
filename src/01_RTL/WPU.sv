/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    WPU.sv
* Project:      tpu-accelerator-lite
* Module:       WPU (Weight Pre-processing Unit)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module WPU (
    input clk,
    input rst_n,
    input [`W_RAW_BW-1:0] weight,
    input [`WMEM_ADDR_W-1:0] wmem_addr_i,
    input wmem_w,
    output logic [`W_BW-1:0] rweight,
    output logic [`WMEM_ADDR_W-1:0] wmem_addr_o
);

    logic [`W_BW-1:0] rweight_c;

    genvar i;
    generate
        for(i = 0; i < `ARRAY_S; i++) begin : ENCODE_LANE
            logic [`W_W-1:0] w;

            always_comb begin
                w = weight[`W_W*i +: `W_W];
                if((&w[7:4]) ^ (|w[7:4])) begin
                    rweight_c[`RW_W*i +: `RW_W] = {1'b1, w[7:4]};
                end
                else begin
                    rweight_c[`RW_W*i +: `RW_W] = {1'b0, w[4:1]};
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rweight <= 0;
            wmem_addr_o <= 0;
        end
        else begin
            if(wmem_w) begin
                rweight <= rweight_c;
                wmem_addr_o <= wmem_addr_i;
            end
            else begin
                rweight <= 0;
                wmem_addr_o <= 0;
            end
        end
    end

endmodule
