/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    RPE.sv
* Project:      tpu-accelerator-lite
* Module:       RPE (Reduced Processing Element)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

// According to Reduced Processing Unit Hardware
module RPE(
    input clk,
    input rst_n,
    input [`RW_W-1:0] weight_i,
    input [`A_W-1:0] activation_i,
    input [`PSUM_W-1:0] psum_i,
    input weight_valid_i,
    output logic [`RW_W-1:0] weight_o,
    output logic weight_valid_o,
    output logic [`A_W-1:0] activation_o,
    output logic [`PSUM_W-1:0] psum_o
);
    logic [`A_W:0] ex_activation_i;
    logic [`PSUM_W-1:0] mac_o;

    always_comb begin
        ex_activation_i = {activation_i,1'b1}; // Activation Extension
        weight_valid_o = weight_valid_i;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            weight_o <= 0;
            activation_o <= 0;
            psum_o <= 0;
        end
        else begin 
            if(weight_valid_i) begin
                weight_o <= weight_i; // Weight pass downward
            end
            else begin
                psum_o <= mac_o;
                activation_o <= activation_i;
            end
        end
    end

    RMAC u_RMAC(
        .activation(ex_activation_i),
        .weight(weight_o),
        .psum_i(psum_i),
        .psum_o(mac_o));

endmodule

module RMAC (
    input [`A_W:0] activation,
    input [`RW_W-1:0] weight,
    input [`PSUM_W-1:0] psum_i,
    output logic [`PSUM_W-1:0] psum_o
);
    localparam MUL_W = (`RW_W - 1) + (`A_W + 1);    // 4-bit * 8-bit = 12-bit
    localparam RES_W = (`W_W) + (`A_W + 1);         // 8-bit * 8-bit = 16-bit
    localparam RESULT_EXTENSION = `PSUM_W - RES_W;

    logic weight_sign_add;
    logic [`RW_W-2:0] weight_i;
    logic [`A_W:0] activation_i;
    logic [MUL_W-1:0] mul_result;
    logic [MUL_W-1:0] sign_result;
    logic [MUL_W:0] shift_result;
    logic [MUL_W:0] msr4_result;
    logic [RES_W-1:0] no_msr4_result;
    logic [RES_W-1:0] result;

    always_comb begin : MAC_BLOCK
        weight_sign_add = weight[`RW_W-2] && (!weight[`RW_W-1]);
        weight_i = (weight[`RW_W-2])? ~(weight[`RW_W-2:0]) + weight_sign_add : weight[`RW_W-2:0];
        activation_i = (activation[`A_W])? ~(activation) + 1 : activation;
        mul_result = activation_i * weight_i;
        sign_result = (activation[`A_W] ^ weight[`RW_W-2])? ~(mul_result) + 1 : mul_result;
        shift_result = {sign_result, 1'b0};
        msr4_result = shift_result + {{5{activation[`A_W]}},activation};
        no_msr4_result = {shift_result,3'b000};
        result = (weight[`RW_W-1])? no_msr4_result : {{3{msr4_result[MUL_W]}},msr4_result};
        psum_o = {{RESULT_EXTENSION{result[RES_W-1]}},result} + psum_i;        
    end

endmodule