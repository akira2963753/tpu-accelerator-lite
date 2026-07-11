/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN (TPU top smoke test)
* Author:       Marco <harry2963753@gmail.com>
*
* Simple single-job smoke pattern : drive one matmul, watch the waveform,
* compare r_data against a plain-matmul golden. Mismatches are $display'd
* (NOT $fatal) so the sim runs to completion and the whole waveform is captured.
* All stimulus is driven on negedge (gate-safe).
******************************************************************************/

`define CLK_PERIOD 10
`define TIMEOUT 5000000

module PATTERN(
    output logic clk,
    output logic rst_n,
    output logic cmd_valid,
    input cmd_ready,
    output logic [`M_W-1:0] dim_m,
    output logic [`K_W-1:0] dim_k,
    output logic [`N_W-1:0] dim_n,
    output logic [`QMULT_W-1:0] quant_mult,
    output logic [`QSHIFT_W-1:0] quant_shift,
    output logic w_valid,
    input w_ready,
    output logic [`W_RAW_BW-1:0] w_data,
    output logic a_valid,
    input a_ready,
    output logic [`A_BW-1:0] a_data,
    output logic r_ready,
    input r_valid,
    input [`A_BW-1:0] r_data,
    input busy,
    input done
);

    //=============================================================
    // ------------------- Job size (match python) ---------------
    //=============================================================
    localparam M = 16;
    localparam K = 16;
    localparam N = 16;
    // per-tensor requant : C_q = saturate( (ReLU(acc) * QMULT) >>> QSHIFT )
    // identity (1,0) reproduces the pre-quant golden ; bump to rescale (keep in sync
    // with tpu_gen.py --qmult / --qshift).
    localparam QMULT  = 1;
    localparam QSHIFT = 0;
    localparam MT = M/`ARRAY_S;
    localparam KT = K/`ARRAY_S;
    localparam NT = N/`ARRAY_S;
    localparam W_BEATS = MT*NT*KT*`ARRAY_S; // weight beats (re-sent per mt)
    localparam A_BEATS = MT*KT*`ARRAY_S;    // activation beats (per mt)
    localparam R_BEATS = MT*NT*`ARRAY_S;    // result beats

    reg [`W_RAW_BW-1:0] mem_w [0:W_BEATS-1];
    reg [`A_BW-1:0] mem_a [0:A_BEATS-1];
    reg [`A_BW-1:0] mem_g [0:R_BEATS-1];
    integer err_cnt;

    //=============================================================
    // ------------------- Clock & Reset --------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    task automatic reset_dut();
        begin
            force clk = 0;
            rst_n = 1;
            cmd_valid = 0; dim_m = 0; dim_k = 0; dim_n = 0;
            quant_mult = 0; quant_shift = 0;
            w_valid = 0; w_data = 0;
            a_valid = 0; a_data = 0;
            r_ready = 0;
            #20 rst_n = 0;
            #20 rst_n = 1;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================
    // valid/ready producer : present beat, if ready it is taken at the next
    // posedge -> advance immediately (ready does not depend on valid).
    task automatic feed_w();
        int i;
        begin
            for (i = 0; i < W_BEATS; ) begin
                w_valid = 1;
                w_data  = mem_w[i];
                if (w_ready) i = i + 1;
                @(negedge clk);
            end
            w_valid = 0;
            w_data  = 0;
        end
    endtask

    task automatic feed_a();
        int i;
        begin
            for (i = 0; i < A_BEATS; ) begin
                a_valid = 1;
                a_data  = mem_a[i];
                if (a_ready) i = i + 1;
                @(negedge clk);
            end
            a_valid = 0;
            a_data  = 0;
        end
    endtask

    task automatic monitor_r();
        int i;
        begin
            r_ready = 1;
            for (i = 0; i < R_BEATS; ) begin
                if (r_valid) begin
                    if (r_data !== mem_g[i]) begin
                        $display("[MISMATCH] result beat %0d : exp=%028h got=%028h",
                            i, mem_g[i], r_data);
                        err_cnt = err_cnt + 1;
                    end
                    i = i + 1;
                end
                @(negedge clk);
            end
        end
    endtask

    task end_task();
        begin
            $display("============================================");
            if (err_cnt == 0)
                $display("  [SUCCESS] ALL %0d RESULT BEATS MATCH !", R_BEATS);
            else
                $display("  [FAIL] %0d / %0d RESULT BEATS MISMATCH", err_cnt, R_BEATS);
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
        $readmemh("../00_TESTBED/pattern/weight_raw.dat", mem_w);
        $readmemh("../00_TESTBED/pattern/activation.dat", mem_a);
        $readmemh("../00_TESTBED/pattern/golden.dat", mem_g);
        err_cnt = 0;
        reset_dut();

        // issue job descriptor
        cmd_valid = 1;
        dim_m = M; dim_k = K; dim_n = N;
        quant_mult = QMULT; quant_shift = QSHIFT;
        while (!cmd_ready) @(negedge clk);
        @(negedge clk);           // descriptor accepted -> LOADING
        cmd_valid = 0;

        // stream loads + collect results concurrently
        fork
            feed_w();
            feed_a();
            monitor_r();
        join_none

        @(posedge done);
        @(negedge clk);
        end_task();
        $finish;
    end

endmodule
