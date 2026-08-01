/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

`define CLK_PERIOD 10
`define TIMEOUT 1000000
`define TILE_NUM 100

module PATTERN(
    output logic clk,
    output logic rst_n,
    output logic [`W_RAW_BW-1:0] weight,
    output logic [`WMEM_ADDR_W-1:0] wmem_addr_i,
    output logic wmem_w,
    input [`W_BW-1:0] rweight,
    input [`CW_BW-1:0] cweight,
    input [`ARRAY_S-1:0] cout_valid,
    input [`ROW_IDX_W-1:0] crow, // row within the 16-row tile, NOT a WMEM address
    input [`WMEM_ADDR_W-1:0] wmem_addr_o
);

    //=============================================================
    // ------------------- Clock & Reset --------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    task automatic reset_dut();
        begin
            force clk = 0;
            rst_n = 1;
            weight = 0;
            wmem_addr_i = 0;
            wmem_w = 0;
            #20 rst_n = 0;
            #20 rst_n = 1;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // --------------------- Weight Generators --------------------
    //=============================================================

    // MSR-4 : top nibble all-equal (0000 or 1111)
    function automatic [`W_W-1:0] rand_msr4();
        reg s;
        reg [3:0] lo;
        begin
            s = $random;
            lo = $random;
            rand_msr4 = {{4{s}}, lo};
        end
    endfunction

    // Non-MSR-4 : top nibble mixed (reject until (&top)^(|top)==1)
    function automatic [`W_W-1:0] rand_non_msr4();
        reg [`W_W-1:0] v;
        begin
            v = $random;
            while (((&v[7:4]) ^ (|v[7:4])) == 1'b0) v = $random;
            rand_non_msr4 = v;
        end
    endfunction

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================

    // Drive one row (16 lanes) then self-check the encoding identity
    task automatic drive_and_check(input [`W_RAW_BW-1:0] wrow, input [`WMEM_ADDR_W-1:0] addr);
        int i;
        reg [`W_W-1:0] w;
        reg signed [`W_W-1:0] weight_exp;    // golden : original weight, LSB fixed to 1
        reg signed [`W_W:0] recon;           // value reconstructed from DUT outputs
        begin
            weight = wrow;
            wmem_addr_i = addr;
            wmem_w = 1;
            @(negedge clk); 

            for (i = 0; i < `ARRAY_S; i = i + 1) begin
                w = wrow[`W_W*i +: `W_W];
                weight_exp = {w[7:1], 1'b1};

                if (cout_valid[i] === 1'b0)
                    // MSR-4 : reduced alone reconstructs the weight
                    recon = $signed({rweight[`RW_W*i +: `RW_W-1], 1'b1});
                else
                    // Non-MSR-4 : reduced (high nibble) + compensation (low part)
                    recon = $signed(rweight[`RW_W*i +: `RW_W-1]) * 16 + $signed({1'b0, cweight[`CW_W*i +: `CW_W-1], 1'b1});

                if (recon !== weight_exp) begin
                    $fatal(1, "[ERROR] : lane=%0d w=%08b msr4=%0b exp=%0d recon=%0d",
                        i, w, ~cout_valid[i], weight_exp, recon);
                end
            end
        end
    endtask

    // Build one 16x16 tile : each column has exactly one Non-MSR-4 across its 16 rows
    task automatic run_tile();
        int r, c;
        int nz_row [0:`ARRAY_S-1];
        reg [`W_W-1:0] tile [0:`ARRAY_S-1][0:`ARRAY_S-1];
        reg [`W_RAW_BW-1:0] wrow;
        begin
            for (c = 0; c < `ARRAY_S; c = c + 1) begin
                nz_row[c] = $urandom_range(0, `ARRAY_S-1);
            end
            for (r = 0; r < `ARRAY_S; r = r + 1) begin
                for (c = 0; c < `ARRAY_S; c = c + 1) begin
                    tile[r][c] = (r == nz_row[c])? rand_non_msr4() : rand_msr4();
                end
            end
            for (r = 0; r < `ARRAY_S; r = r + 1) begin
                wrow = 0;
                for (c = 0; c < `ARRAY_S; c = c + 1) begin
                    wrow[`W_W*c +: `W_W] = tile[r][c];
                end
                drive_and_check(wrow, r[`WMEM_ADDR_W-1:0]);
            end
        end
    endtask

    task end_task();
        begin
            $display("=============================================================");
            $display("  [SUCCESS] ALL PATTERN & ASSERTION PASS !  (TEST NUM : %0d) ", `TILE_NUM);
            $display("=============================================================");
        end
    endtask

    //=============================================================
    // ------------------------- Timing Watchdog ------------------
    //=============================================================
    initial begin
        #(`TIMEOUT);
        $fatal(1, "[TIMEOUT] : Simulation time exceeded watchdog limit.");
    end

    //=============================================================
    // ------------------------- Main Flow ------------------------
    //=============================================================
    initial begin
        reset_dut();
        for (int t = 0; t < `TILE_NUM; t = t + 1) begin
            run_tile();
        end
        #100 end_task();
        $finish;
    end

endmodule
