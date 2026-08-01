/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    WPU.sv
* Project:      tpu-accelerator-lite
* Module:       WPU (Weight Pre-processing Unit)
* Author:       Marco <harry2963753@gmail.com>
* Veri:         PASS
*
******************************************************************************/

module WPU (
    input clk,                                
    input rst_n,                                
    input [`W_RAW_BW-1:0] weight,               // complete weight 
    input [`WMEM_ADDR_W-1:0] wmem_addr_i,       // wmem waddr in
    input wmem_w,                               // wmem wenable
    output logic [`W_BW-1:0] rweight,           // reduced weight
    output logic [`CW_BW-1:0] cweight,          // compensated weight
    output logic [`ARRAY_S-1:0] cout_valid,     // compensated valid
    output logic [`ROW_IDX_W-1:0] crow,         // compensated row inedx -> "data_setup"
    output logic [`WMEM_ADDR_W-1:0] wmem_addr_o // wmem waddr out
);

    logic [`W_BW-1:0] rweight_c;    // reduced weight
    logic [`CW_BW-1:0] cweight_c;   // compensated weight 
    logic [`ARRAY_S-1:0] cvalid_c;  // compensated valid

    genvar i;
    generate
        for(i = 0; i < `ARRAY_S; i = i + 1) begin : ENCODE_LANE
            logic [`W_W-1:0] w;
            logic non_msr4;
            always_comb begin
                w = weight[`W_W*i +: `W_W];
                non_msr4 = (&w[7:4]) ^ (|w[7:4]);
                rweight_c[`RW_W*i +: `RW_W] = (non_msr4)? {1'b1,w[7:4]} : {1'b0,w[4:1]};
                cweight_c[`CW_W*i +: `CW_W] = {w[`W_W-1],w[3:1]};
                cvalid_c[i] = non_msr4;
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rweight <= 0;
            cweight <= 0;
            cout_valid <= 0;
            crow <= 0;
            wmem_addr_o <= 0;
        end
        else begin
            if(wmem_w) begin
                rweight <= rweight_c;
                cweight <= cweight_c;
                cout_valid <= cvalid_c;
                crow <= wmem_addr_i[`ROW_IDX_W-1:0]; // row within the 16-row tile
                wmem_addr_o <= wmem_addr_i;
            end
            else begin
                rweight <= 0;
                cweight <= 0;
                cout_valid <= 0;
                crow <= 0;
                wmem_addr_o <= 0;
            end
        end
    end

endmodule
