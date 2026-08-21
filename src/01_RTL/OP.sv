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
    input signed [`QMULT_W-1:0] quant_mult,
    input [`QSHIFT_W-1:0] quant_shift,
    input [`ARRAY_S*`ACC_W-1:0] acc_i,
    output logic [`A_BW-1:0] data_o
);

    localparam signed [`ACC_W-1:0] SAT_MAX = (1 <<< (`A_W-1)) - 1;

    always_comb begin
        for(int c = 0; c < `ARRAY_S; c++) begin
            logic signed [`ACC_W-1:0] acc_c;
            logic signed [`ACC_W-1:0] relu_c;
            logic signed [`ACC_W+`QMULT_W-1:0] prod_c;
            logic signed [`ACC_W+`QMULT_W-1:0] q_c;
            acc_c = $signed(acc_i[`ACC_W*c +: `ACC_W]);
            relu_c = acc_c[`ACC_W-1] ? '0 : acc_c;
            prod_c = relu_c * quant_mult;
            q_c = prod_c >>> quant_shift;
            data_o[`A_W*c +: `A_W] = (q_c > SAT_MAX) ? SAT_MAX[`A_W-1:0] : q_c[`A_W-1:0];
        end
    end

endmodule
