/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    OP.sv
* Project:      tpu-accelerator-lite
* Module:       OP (Output Processing, ReLU + QUANTIZATION)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module OP (
    input signed [`QMULT_W-1:0] quant_mult, // per-tensor requant multiplier M0
    input [`QSHIFT_W-1:0] quant_shift,       // per-tensor requant right-shift n
    input [`ARRAY_S*`ACC_W-1:0] acc_i,       // one ACC row : 16 x ACC_W (signed)
    output logic [`A_BW-1:0] data_o          // quantized activations : 16 x A_W
);

    // per-tensor requant : C_q = saturate( (ReLU(acc) * M0) >>> n ) , truncating (no round).
    // M0/n fold S_a*S_w/S_c ; host precomputes them. ReLU first (M0 >= 0 keeps sign).
    localparam signed [`ACC_W-1:0] SAT_MAX = (1 <<< (`A_W-1)) - 1; // max positive A_W signed = 63

    always_comb begin
        for(int c = 0; c < `ARRAY_S; c++) begin
            logic signed [`ACC_W-1:0] acc_c;
            logic signed [`ACC_W-1:0] relu_c;
            logic signed [`ACC_W+`QMULT_W-1:0] prod_c;
            logic signed [`ACC_W+`QMULT_W-1:0] q_c;
            acc_c = $signed(acc_i[`ACC_W*c +: `ACC_W]);
            relu_c = acc_c[`ACC_W-1] ? '0 : acc_c; // ReLU : clamp negatives to 0
            prod_c = relu_c * quant_mult; // requant multiply (M0), signed
            q_c = prod_c >>> quant_shift; // requant right-shift (truncate, >=0)
            data_o[`A_W*c +: `A_W] = (q_c > SAT_MAX) ? SAT_MAX[`A_W-1:0] : q_c[`A_W-1:0]; // saturate to A_W
        end
    end

endmodule
