/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    ACC.sv
* Project:      tpu-accelerator-lite
* Module:       ACC (Accumulator)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module ACC (
    input clk,
    input rst_n,
    input acc_clr,
    input [`PSUM_BW-1:0] psum_i,
    input [`ARRAY_S-1:0] acc_wr_en,
    input [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row,
    input acc_rd_en,
    input [`ROW_IDX_W-1:0] acc_rd_row,
    output logic [`ARRAY_S*`ACC_W-1:0] acc_o
);

    logic signed [`ACC_W-1:0] acc_mem [0:`ARRAY_S-1][0:`ARRAY_S-1];
    logic signed [`ACC_W-1:0] psum_ext [0:`ARRAY_S-1];

    always_comb begin
        for(int c = 0; c < `ARRAY_S; c++) begin
            psum_ext[c] = {{(`ACC_W-`PSUM_W){psum_i[`PSUM_W*(c+1)-1]}}, psum_i[`PSUM_W*c +: `PSUM_W]};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for(int c = 0; c < `ARRAY_S; c++)
                for(int r = 0; r < `ARRAY_S; r++) acc_mem[c][r] <= 0;
        end
        else if(acc_clr) begin
            for(int c = 0; c < `ARRAY_S; c++)
                for(int r = 0; r < `ARRAY_S; r++) acc_mem[c][r] <= 0;
        end
        else begin
            for(int c = 0; c < `ARRAY_S; c++) begin
                if(acc_wr_en[c]) acc_mem[c][acc_wr_row[`ROW_IDX_W*c +: `ROW_IDX_W]] <= acc_mem[c][acc_wr_row[`ROW_IDX_W*c +: `ROW_IDX_W]] + psum_ext[c];
            end
        end
    end

    always_comb begin
        acc_o = '0;
        if(acc_rd_en)
            for(int c = 0; c < `ARRAY_S; c++) acc_o[`ACC_W*c +: `ACC_W] = acc_mem[c][acc_rd_row];
    end

endmodule
