/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    CPE.sv
* Project:      tpu-accelerator-lite
* Module:       CPE (Compensated Processing Element)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module CPE (
    input clk,
    input rst_n,
    input [`CW_W-1:0] cweight_i,   
    input cweight_valid_i,                 
    input [`A_W-1:0] activation_i,          
    input activation_valid_i,                
    output logic [`CPSUM_W-1:0] cpsum_o       
);
    logic [`CW_W-1:0] cweight_r;        
    logic [`A_W:0] ex_activation_i;     
    logic [`CW_W:0] ex_weight_i;          
    logic [`CPSUM_W-1:0] mac_o;

    always_comb begin
        ex_activation_i = {activation_i, 1'b1}; 
        ex_weight_i = {cweight_r, 1'b1}; 
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            cweight_r <= 0;
            cpsum_o <= 0;
        end
        else begin
            if(cweight_valid_i) cweight_r <= cweight_i;
            if(activation_valid_i) cpsum_o <= mac_o;
        end
    end

    CMAC u_CMAC(
        .activation(ex_activation_i),
        .weight(ex_weight_i),
        .cpsum_o(mac_o));

endmodule

module CMAC (
    input signed [`A_W:0] activation,
    input signed [`CW_W:0] weight,     
    output logic signed [`CPSUM_W-1:0] cpsum_o
);
    always_comb cpsum_o = activation * weight;

endmodule
