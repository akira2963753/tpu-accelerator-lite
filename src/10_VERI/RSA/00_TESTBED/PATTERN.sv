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
`define PAT_NUM 500
`define CAP_LAT 15

module PATTERN(
    output logic clk,
    output logic rst_n,
    output logic weight_valid,
    output logic [`W_BW-1:0] weight,
    output logic [`A_BW-1:0] activation,
    input [`PSUM_BW-1:0] psum
);

    //=============================================================
    // ------- Parameters, Integers & Internal Signals ------------
    //=============================================================
    localparam M = `ARRAY_S;  
    localparam PW = `PSUM_W;

    reg [`W_BW-1:0] mem_w [0:`PAT_NUM*`ARRAY_S-1];  // reduced weight rows (16x5-bit)
    reg [`A_BW-1:0] mem_a [0:`PAT_NUM*M-1];         // activation rows (16x7-bit)
    reg [`ARRAY_S*PW-1:0] mem_g [0:`PAT_NUM*M-1];   // golden psum rows (16x20-bit)

    //=============================================================
    // ------------------- Clock & Reset --------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    task automatic reset_dut();
        begin
            force clk = 0;
            rst_n = 1;
            weight_valid = 0;
            weight = 0;
            activation = 0;
            #20 rst_n = 0;
            #20 rst_n = 1;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================

    // Weight-stationary Pre-load
    task automatic load_weight(input int p);
        int c;
        begin
            weight_valid = 1;
            for (c = 0; c < `ARRAY_S; c = c + 1) begin
                weight = mem_w[p*`ARRAY_S + (`ARRAY_S-1-c)];
                @(negedge clk);
            end
            weight_valid = 0;
            weight = 0;
        end
    endtask

    // Activation Load
    task automatic run_pattern(input int p);
        int cyc, r, c, m;
        reg [`A_BW-1:0] act_bus;
        reg [PW-1:0] exp_psum, act_psum;
        begin
            for (cyc = 0; cyc <= (M-1) + (`ARRAY_S-1) + `CAP_LAT; cyc = cyc + 1) begin
                act_bus = 0;
                for (r = 0; r < `ARRAY_S; r = r + 1) begin
                    m = cyc - r;
                    if (m >= 0 && m < M)
                        act_bus[`A_W*r +: `A_W] = mem_a[p*M + m][`A_W*r +: `A_W];
                end
                activation = act_bus;
                @(negedge clk);

                for (c = 0; c < `ARRAY_S; c = c + 1) begin
                    m = cyc - `CAP_LAT - c;
                    if (m >= 0 && m < M) begin
                        exp_psum = mem_g[p*M + m][PW*c +: PW];
                        act_psum = psum[PW*c +: PW];
                        if (act_psum !== exp_psum) begin
                            $fatal(1, "[ERROR] : p=%0d m=%0d c=%0d exp=%05h act=%05h",
                                p, m, c, exp_psum, act_psum);
                        end
                    end
                end
            end
        end
    endtask

    task end_task();
        begin
            $display("============================================");
            $display("  [SUCCESS] ALL PATTERN & ASSERTION PASS !  ");
            $display("============================================");
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
        $readmemh("../00_TESTBED/pattern/weight_reduced.dat", mem_w);
        $readmemh("../00_TESTBED/pattern/activation.dat", mem_a);
        $readmemh("../00_TESTBED/pattern/golden.dat", mem_g);
        reset_dut();
        for (int p = 0; p < `PAT_NUM; p = p + 1) begin
            load_weight(p);
            run_pattern(p);
        end
        #100 end_task();
        $finish;
    end

endmodule
