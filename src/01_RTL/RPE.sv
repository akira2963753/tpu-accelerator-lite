/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    RPE.sv
* Project:      tpu-accelerator-lite
* Module:       RPE (Reduced Processing Element)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module RPE (
    input clk,
    input rst_n,
    input [`RW_W-1:0] weight_i,
    input [`A_W-1:0] activation_i,
    input activation_valid_i,
    input [`PSUM_W-1:0] psum_i,
    input weight_valid_i,
    output logic [`RW_W-1:0] weight_o,
    output logic weight_valid_o,
    output logic [`A_W-1:0] activation_o,
    output logic activation_valid_o,
    output logic [`PSUM_W-1:0] psum_o
);

    logic [`PSUM_W-1:0] mac_o;

    always_comb begin : WEIGHT_VALID
        weight_valid_o = weight_valid_i;
    end

    always_ff @(posedge clk or negedge rst_n) begin : RPE_REG
        if(!rst_n) begin
            weight_o <= 0;
            activation_o <= 0;
            activation_valid_o <= 0;
            psum_o <= 0;
        end
        else begin
            if(weight_valid_i) begin
                weight_o <= weight_i;
                activation_valid_o <= 0;
            end
            else begin
                psum_o <= mac_o;
                activation_o <= activation_i;
                activation_valid_o <= activation_valid_i;
            end
        end
    end

    RMAC u_RMAC (
        .activation(activation_i),
        .activation_valid(activation_valid_i),
        .weight(weight_o),
        .psum_i(psum_i),
        .psum_o(mac_o)
    );

endmodule

module RMAC (
    input [`A_W-1:0] activation,
    input activation_valid,
    input [`RW_W-1:0] weight,
    input [`PSUM_W-1:0] psum_i,
    output logic [`PSUM_W-1:0] psum_o
);
    logic signed [`RW_W-1:0] weight_5b;
    logic signed [`A_W:0] activation_5b;
    logic signed [`RPE_MUL_W-1:0] product;
    logic signed [`RPE_RES_W-1:0] product_ext;
    logic signed [`RPE_RES_W-1:0] contribution;
    logic signed [`PSUM_W-1:0] contribution_ext;

    always_comb begin : MAC_BLOCK
        weight_5b = $signed({weight[`RW_W-2:0], 1'b1});
        activation_5b = $signed({activation, 1'b1});
        product = weight_5b * activation_5b;
        product_ext = {{(`RPE_RES_W-`RPE_MUL_W){product[`RPE_MUL_W-1]}}, product};
        contribution = (weight[`RW_W-1])? product_ext <<< 6 : product_ext <<< 3;
        contribution_ext = {{(`PSUM_W-`RPE_RES_W){contribution[`RPE_RES_W-1]}}, contribution};
        psum_o = (activation_valid)? $signed(psum_i) + contribution_ext : $signed(psum_i);
    end

endmodule
