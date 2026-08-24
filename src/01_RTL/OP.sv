/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    OP.sv
* Project:      tpu-accelerator-lite
* Module:       OP (Output Processing)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module OP (
    input signed [`QMULT_W-1:0] quant_mult,
    input [`QSHIFT_W-1:0] quant_shift,
    input [`ACT_MODE_W-1:0] act_mode,
    input [`ARRAY_S*`ACC_W-1:0] acc_i,
    output logic [`A_BW-1:0] data_o
);

    localparam CALC_W = `ACC_W + `QMULT_W;
    localparam signed [CALC_W-1:0] SAT_MAX = (1 <<< (`A_W-1)) - 1;
    localparam signed [CALC_W-1:0] SAT_MIN = -(1 <<< (`A_W-1));

    always_comb begin : OUTPUT_PROCESS
        for(int c = 0; c < `ARRAY_S; c++) begin
            logic signed [`ACC_W-1:0] acc_c;
            logic signed [`ACC_W-1:0] act_c;
            logic signed [CALC_W-1:0] prod_c;
            logic signed [CALC_W-1:0] q_c;
            acc_c = $signed(acc_i[`ACC_W*c +: `ACC_W]);
            act_c = (act_mode == `ACT_RELU && acc_c[`ACC_W-1])? '0 : acc_c;
            prod_c = act_c * quant_mult;
            q_c = prod_c >>> quant_shift;

            if(act_mode == `ACT_RELU) begin
                if(q_c < 0) data_o[`A_W*c +: `A_W] = 0;
                else if(q_c > SAT_MAX) data_o[`A_W*c +: `A_W] = SAT_MAX[`A_W-1:0];
                else data_o[`A_W*c +: `A_W] = q_c[`A_W-1:0];
            end
            else begin
                if(q_c > SAT_MAX) data_o[`A_W*c +: `A_W] = SAT_MAX[`A_W-1:0];
                else if(q_c < SAT_MIN) data_o[`A_W*c +: `A_W] = SAT_MIN[`A_W-1:0];
                else data_o[`A_W*c +: `A_W] = q_c[`A_W-1:0];
            end
        end
    end

endmodule
