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
    input acc_clr, // clear whole block at (mt,nt) start

    // datapath inputs (packed, one slice per column)
    input [`PSUM_BW-1:0] psum_i, // RSA : ARRAY_S x PSUM_W
    input [`ARRAY_S*`CPSUM_W-1:0] cpsum_i, // CSA : ARRAY_S x CPSUM_W
    input [`ARRAY_S-1:0] cout_valid, // per-column compensation valid (from Data_Setup)

    // psum accumulate control (from TSC ; en + row are per-column, de-skew staggered)
    input [`ARRAY_S-1:0] acc_wr_en,
    input [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row,

    // compensation add control (from TSC ; window pulse + row = output row m)
    input cacc_en,
    input [`ROW_IDX_W-1:0] cacc_wr_row,

    // read-out (OUT phase : sweep row 0..ARRAY_S-1)
    input acc_rd_en,
    input [`ROW_IDX_W-1:0] acc_rd_row,
    output logic [`ARRAY_S*`ACC_W-1:0] acc_o
);

    // [column][row] accumulate memory
    logic signed [`ACC_W-1:0] acc_mem [0:`ARRAY_S-1][0:`ARRAY_S-1];

    // sign-extended per-column input slices
    logic signed [`ACC_W-1:0] psum_ext [0:`ARRAY_S-1];
    logic signed [`ACC_W-1:0] cpsum_ext [0:`ARRAY_S-1];

    always_comb begin
        for(int c = 0; c < `ARRAY_S; c++) begin
            psum_ext[c] = $signed(psum_i[`PSUM_W*c +: `PSUM_W]);
            cpsum_ext[c] = $signed(cpsum_i[`CPSUM_W*c +: `CPSUM_W]);
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
                if(cacc_en && cout_valid[c]) acc_mem[c][cacc_wr_row] <= acc_mem[c][cacc_wr_row] + cpsum_ext[c];
                if(acc_wr_en[c]) acc_mem[c][acc_wr_row[`ROW_IDX_W*c +: `ROW_IDX_W]] <= acc_mem[c][acc_wr_row[`ROW_IDX_W*c +: `ROW_IDX_W]] + psum_ext[c];
            end
        end
    end

    // read-out one row (16 columns) : combinational register-file read (no latency)
    // so r_data lines up with r_valid in the OUT phase (no 1-cycle output skew).
    always_comb begin
        acc_o = '0;
        if(acc_rd_en)
            for(int c = 0; c < `ARRAY_S; c++) acc_o[`ACC_W*c +: `ACC_W] = acc_mem[c][acc_rd_row];
    end

endmodule
