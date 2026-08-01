/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    Data_Setup.sv
* Project:      tpu-accelerator-lite
* Module:       Data_Setup
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module Data_Setup (
    input clk,
    input rst_n,
    input [`A_BW-1:0] activation_i,               // 16 x 7-bit activation row (packed, un-skewed)
    input skew_en,                                // calculation phase : advance skew pipeline
    input comp_clr,                               // clear compensation state at each tile load
    input [`ARRAY_S-1:0] cout_valid,              // per-column Non-MSR-4 mask (from WPU)
    input [`ROW_IDX_W-1:0] crow,                  // compensation row index, shared by all lanes (from WPU)
    output logic [`A_BW-1:0] activation_o,        // 45-degree skewed activation to RSA
    output logic [`A_BW-1:0] activation_cout_o,   // compensation activation to CSA (one per column)
    output logic [`ARRAY_S-1:0] cout_valid_o      // per-column compensation valid
);

    // declarate directly S * S skew registers, tool would be remove useless registers
    logic [`A_W-1:0] skew [0:`ARRAY_S-1][0:`ARRAY_S-1];
    logic [`ROW_IDX_W-1:0] comp_row [0:`ARRAY_S-1];
    logic [`ARRAY_S-1:0] comp_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            // k starts at 0 : skew[r][0] is the live input stage and skew[0][0] IS
            // activation_o lane 0, so leaving it un-reset drives X into the array.
            for(int r = 0; r < `ARRAY_S; r++)
                for(int k = 0; k <= r; k++)
                    skew[r][k] <= 0;
        end
        else if(skew_en) begin
            for(int r = 0; r < `ARRAY_S; r++) begin
                skew[r][0] <= activation_i[`A_W*r +: `A_W]; // load external activation input into skew
                for(int k = 1; k <= r; k++) skew[r][k] <= skew[r][k-1]; // skew pipelining
            end
        end
    end

    // activation output 
    always_comb for(int r = 0; r < `ARRAY_S; r++) activation_o[`A_W*r +: `A_W] = skew[r][r];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            comp_valid <= 0;
            for(int j = 0; j < `ARRAY_S; j++) comp_row[j] <= 0;
        end
        else if(comp_clr) begin // new tile load : drop previous tile's compensation state
            comp_valid <= 0;
            for(int j = 0; j < `ARRAY_S; j++) comp_row[j] <= 0;
        end
        else begin
            for(int j = 0; j < `ARRAY_S; j++) begin
                if(cout_valid[j]) begin // load compensation row index
                    comp_row[j] <= crow;
                    comp_valid[j] <= 1'b1;
                end
            end
        end
    end

    always_comb begin
        cout_valid_o = comp_valid;
        for(int j = 0; j < `ARRAY_S; j++) begin
            if(comp_valid[j]) activation_cout_o[`A_W*j +: `A_W] = activation_i[`A_W*comp_row[j] +: `A_W];
            else activation_cout_o[`A_W*j +: `A_W] = '0;
        end
    end

endmodule
