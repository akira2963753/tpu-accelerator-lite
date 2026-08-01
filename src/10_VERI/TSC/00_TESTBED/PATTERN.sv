/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN (TSC standalone control check)
* Author:       Marco <harry2963753@gmail.com>
*
* Fast (~1 s) sanity check of the controller alone -- run this before spending
* minutes on the full top-level suite. It drives a job at full throughput and
* self-checks, without any datapath :
*   - external beat counts    : w = M/16 * N/16 * K/16 * 16, r = M/16 * N/16 * 16
*   - UB streaming window     : read n must fetch the slot written at beat n
*                               (proves the kt[0] half select and the 16-row
*                                write/read order line up)
*   - OBUF fusion addressing  : producer writes [nt*16+m], consumer reads
*                               [kt*16+m] -- the sequences must be identical
*   - a fused job must never assert a_ready (no host activation traffic)
******************************************************************************/

`define CLK_PERIOD 10
`define TIMEOUT 50000000

module PATTERN(
    output logic clk,
    output logic rst_n,
    output logic cmd_valid,
    output logic [`M_W-1:0] dim_m,
    output logic [`K_W-1:0] dim_k,
    output logic [`N_W-1:0] dim_n,
    output logic [`QMULT_W-1:0] quant_mult,
    output logic [`QSHIFT_W-1:0] quant_shift,
    output logic fuse_in,
    output logic fuse_out,
    output logic w_valid,
    output logic a_valid,
    output logic r_ready,
    input cmd_ready,
    input busy,
    input done,
    input w_ready,
    input a_ready,
    input r_valid,
    input [`AMEM_ADDR_W-1:0] ub_addr_w,
    input [`AMEM_ADDR_W-1:0] ub_addr_r,
    input ub_en_w,
    input ub_en_r,
    input [`OBUF_ADDR_W-1:0] obuf_addr_w,
    input [`OBUF_ADDR_W-1:0] obuf_addr_r,
    input obuf_en_w,
    input obuf_en_r,
    input act_sel
);

    localparam MAXSEQ = 4096;

    int w_beats, a_beats, r_beats;
    int ub_wr, ub_rd, ob_wr, ob_rd;
    int ub_rd_expect;
    int errors;
    int ub_shadow  [0:(1<<`AMEM_ADDR_W)-1];
    int ob_wr_seq  [0:MAXSEQ-1];
    int ob_rd_seq  [0:MAXSEQ-1];

    //=============================================================
    // ------------------------ Clock -----------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    //=============================================================
    // ----------------------- Scoreboard -------------------------
    //=============================================================
    task automatic clear_sb();
        begin
            w_beats = 0; a_beats = 0; r_beats = 0;
            ub_wr = 0; ub_rd = 0; ob_wr = 0; ob_rd = 0;
            ub_rd_expect = 0;
            for (int i = 0; i < (1<<`AMEM_ADDR_W); i++) ub_shadow[i] = -1;
        end
    endtask

    always @(posedge clk) if (rst_n) begin
        if (w_valid && w_ready) w_beats++;
        if (a_valid && a_ready) a_beats++;
        if (r_valid && r_ready) r_beats++;
        if (ub_en_w) begin
            ub_shadow[ub_addr_w] = ub_wr;   // tag the slot with its beat number
            ub_wr++;
        end
        if (ub_en_r) begin
            if (ub_shadow[ub_addr_r] !== ub_rd_expect) begin
                $display("  [ERROR] UB read %0d @addr %0d holds beat %0d (expected %0d)",
                    ub_rd, ub_addr_r, ub_shadow[ub_addr_r], ub_rd_expect);
                errors++;
            end
            ub_rd++;
            ub_rd_expect++;
        end
        if (obuf_en_w) begin
            if (ob_wr < MAXSEQ) ob_wr_seq[ob_wr] = obuf_addr_w;
            ob_wr++;
        end
        if (obuf_en_r) begin
            if (ob_rd < MAXSEQ) ob_rd_seq[ob_rd] = obuf_addr_r;
            ob_rd++;
        end
        if (act_sel && a_ready) begin
            $display("  [ERROR] a_ready asserted while act_sel=1 (fused job)");
            errors++;
        end
    end

    //=============================================================
    // ------------------------- Reset ----------------------------
    //=============================================================
    task automatic reset_dut();
        begin
            force clk = 0;
            rst_n = 1;
            cmd_valid = 0;
            dim_m = 0; dim_k = 0; dim_n = 0;
            quant_mult = 1; quant_shift = 0;
            fuse_in = 0; fuse_out = 0;
            w_valid = 1; a_valid = 1; r_ready = 1;   // never back-pressure
            #20 rst_n = 0;
            #20 rst_n = 1;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================
    task automatic run(input int M, input int K, input int N,
                       input bit fin, input bit fout, input string nm);
        int MT, KT, NT, exp_w, exp_a, exp_r, exp_ub, exp_ob;
        begin
            MT = M/`ARRAY_S; KT = K/`ARRAY_S; NT = N/`ARRAY_S;
            exp_w  = MT*NT*KT*`ARRAY_S;
            exp_a  = fin ? 0 : MT*NT*KT*`ARRAY_S;
            exp_r  = MT*NT*`ARRAY_S;
            exp_ub = fin ? 0 : MT*NT*KT*`ARRAY_S;
            exp_ob = fin ? MT*NT*KT*`ARRAY_S : (fout ? MT*NT*`ARRAY_S : 0);
            clear_sb();

            dim_m = M; dim_k = K; dim_n = N;
            fuse_in = fin; fuse_out = fout;
            cmd_valid = 1;
            while (!cmd_ready) @(negedge clk);
            @(negedge clk);
            cmd_valid = 0;
            while (!done) @(negedge clk);
            @(negedge clk);
            fuse_in = 0; fuse_out = 0;

            $display("  %-20s w=%0d/%0d a=%0d/%0d r=%0d/%0d ub=%0d/%0d ob_wr=%0d ob_rd=%0d",
                nm, w_beats, exp_w, a_beats, exp_a, r_beats, exp_r,
                ub_rd, exp_ub, ob_wr, ob_rd);

            if (w_beats != exp_w)  begin $display("  [ERROR] %s weight beats", nm); errors++; end
            if (a_beats != exp_a)  begin $display("  [ERROR] %s activation beats", nm); errors++; end
            if (r_beats != exp_r)  begin $display("  [ERROR] %s result beats", nm); errors++; end
            if (ub_wr   != exp_ub) begin $display("  [ERROR] %s UB writes", nm); errors++; end
            if (ub_rd   != exp_ub) begin $display("  [ERROR] %s UB reads", nm); errors++; end

            if (fout) begin
                if (ob_wr != MT*NT*`ARRAY_S) begin $display("  [ERROR] %s OBUF writes", nm); errors++; end
                // single-mt producer : [nt*16+m] must sweep 0..N-1 in order
                for (int i = 0; i < ob_wr; i++)
                    if (ob_wr_seq[i] != i) begin
                        $display("  [ERROR] %s OBUF wr addr[%0d]=%0d expected %0d",
                            nm, i, ob_wr_seq[i], i);
                        errors++;
                    end
            end
            else if (ob_wr != 0) begin
                $display("  [ERROR] %s wrote OBUF without fuse_out", nm); errors++;
            end

            if (fin) begin
                if (ob_rd != exp_ob) begin $display("  [ERROR] %s OBUF reads", nm); errors++; end
                // single-block consumer : [kt*16+m] must replay the producer's
                // write addresses exactly, else fusion feeds the wrong rows
                for (int i = 0; i < ob_rd; i++)
                    if (ob_rd_seq[i] != i) begin
                        $display("  [ERROR] %s OBUF rd addr[%0d]=%0d expected %0d",
                            nm, i, ob_rd_seq[i], i);
                        errors++;
                    end
            end
            else if (ob_rd != 0) begin
                $display("  [ERROR] %s read OBUF without fuse_in", nm); errors++;
            end
        end
    endtask

    //=============================================================
    // ------------------------- Timing Watchdog ------------------
    //=============================================================
    initial begin
        #(`TIMEOUT);
        $fatal(1, "[ERROR] : Simulation time exceeded watchdog limit.");
    end

    //=============================================================
    // ------------------------- Main Flow ------------------------
    //=============================================================
    initial begin
        errors = 0;
        clear_sb();
        $display("======================================");
        $display("  TSC CONTROL CHECK");
        $display("======================================");
        reset_dut();
        run(  16,   16,   16, 0, 0, "16x16x16");
        run(  32,   48,   32, 0, 0, "32x48x32");
        run(  64,   16,   32, 0, 0, "64x16x32");
        run(  16, 1024,   16, 0, 0, "16x1024x16 bigK");
        run(  16, 4096,   16, 0, 0, "16x4096x16 maxK");
        run( 128,   16,   16, 0, 0, "128x16x16 bigM");
        run(  16,   16,  256, 0, 0, "16x16x256 bigN");
        run(  16,   16,   32, 0, 1, "fuse_src");
        run(  16,   32,   16, 1, 0, "fuse_dst K=32");
        run(  16,  128,   16, 1, 0, "fuse_dst K=128");
        $display("======================================");
        if (errors == 0) $display("  [SUCCESS] TSC CONTROL CHECK PASS");
        else             $display("  [FAILURE] %0d ERRORS", errors);
        $display("======================================");
        $finish;
    end

endmodule
