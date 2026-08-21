/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN (TSC standalone serial-command control check)
* Author:       Marco <harry2963753@gmail.com>
*
* TSC standalone control check.
******************************************************************************/

`define CLK_PERIOD 10
`define TIMEOUT 50000000

module PATTERN(
    output logic                   clk,
    output logic                   rst_n,
    output logic                   cmd_valid,
    output logic [`CMD_DESC_W-1:0] cmd_desc,
    output logic                   w_valid,
    output logic                   a_valid,
    output logic                   r_ready,
    input  logic                   cmd_ready,
    input  logic                   busy,
    input  logic                   done,
    input  logic                   w_ready,
    input  logic                   a_ready,
    input  logic                   r_valid,
    input  logic [`AMEM_ADDR_W-1:0] ub_addr_w,
    input  logic [`AMEM_ADDR_W-1:0] ub_addr_r,
    input  logic                   ub_en_w,
    input  logic                   ub_en_r,
    input  logic [`OBUF_ADDR_W-1:0] obuf_addr_w,
    input  logic [`OBUF_ADDR_W-1:0] obuf_addr_r,
    input  logic                   obuf_en_w,
    input  logic                   obuf_en_r,
    input  logic                   act_sel,
    input  logic                   acc_clr
);

    localparam int MAXSEQ = 4096;

    int w_beats, a_beats, r_beats;
    int ub_wr, ub_rd, ob_wr, ob_rd;
    int ub_rd_expect;
    int errors;
    int acc_clr_total;
    int acc_init_cmds;
    int acc_noninit_cmds;
    int current_cmd_acc_clr;
    logic current_cmd_is_gemm;
    logic current_cmd_acc_init;
    int ub_shadow [0:(1 << `AMEM_ADDR_W)-1];
    int ob_wr_seq [0:MAXSEQ-1];
    int ob_rd_seq [0:MAXSEQ-1];

    //=============================================================
    // ------------------------ Clock -----------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    //=============================================================
    // ----------------------- Scoreboard -------------------------
    //=============================================================
    task automatic clear_sb();
        begin
            w_beats     = 0;
            a_beats     = 0;
            r_beats     = 0;
            ub_wr       = 0;
            ub_rd       = 0;
            ob_wr       = 0;
            ob_rd       = 0;
            ub_rd_expect = 0;
            acc_clr_total       = 0;
            acc_init_cmds       = 0;
            acc_noninit_cmds    = 0;
            current_cmd_acc_clr = 0;
            current_cmd_is_gemm = 0;
            current_cmd_acc_init = 0;
            for (int i = 0; i < (1 << `AMEM_ADDR_W); i++) begin
                ub_shadow[i] = -1;
            end
        end
    endtask

    always @(posedge clk) if (rst_n) begin
        if (w_valid && w_ready) w_beats++;
        if (a_valid && a_ready) a_beats++;
        if (r_valid && r_ready) r_beats++;

        if (ub_en_w) begin
            ub_shadow[ub_addr_w] = ub_wr;
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
            $display("  [ERROR] a_ready asserted while act_sel=1 (fused GEMM)");
            errors++;
        end
        if (acc_clr) begin
            acc_clr_total++;
            current_cmd_acc_clr++;
            if (!(current_cmd_is_gemm && current_cmd_acc_init)) begin
                $display("  [ERROR] acc_clr asserted outside an ACC_INIT GEMM");
                errors++;
            end
        end
    end

    //=============================================================
    // ---------------- Descriptor / Command Tasks ----------------
    //=============================================================
    function automatic logic [`CMD_DESC_W-1:0] build_cmd_desc(
        input logic [`CMD_OP_W-1:0]    opcode,
        input logic                     acc_init,
        input logic                     acc_final,
        input logic                     fuse_in,
        input logic                     fuse_out,
        input int                       mt,
        input int                       kt,
        input int                       nt
    );
        logic [`CMD_DESC_W-1:0] desc;
        begin
            desc = '0;
            desc[`CMD_OP_LSB +: `CMD_OP_W]    = opcode;
            desc[`CMD_ACC_INIT_BIT]            = acc_init;
            desc[`CMD_ACC_FINAL_BIT]           = acc_final;
            desc[`CMD_FUSE_IN_BIT]             = fuse_in;
            desc[`CMD_FUSE_OUT_BIT]            = fuse_out;
            desc[`CMD_QMULT_LSB +: `QMULT_W]   = `QMULT_W'(1);
            desc[`CMD_QSHIFT_LSB +: `QSHIFT_W] = '0;
            desc[`CMD_MT_LSB +: `CMD_MT_W]     = mt[`CMD_MT_W-1:0];
            desc[`CMD_KT_LSB +: `CMD_KT_W]     = kt[`CMD_KT_W-1:0];
            desc[`CMD_NT_LSB +: `CMD_NT_W]     = nt[`CMD_NT_W-1:0];
            build_cmd_desc = desc;
        end
    endfunction

    task automatic issue_cmd(
        input logic [`CMD_DESC_W-1:0] desc,
        input string                  label
    );
        logic [`CMD_OP_W-1:0] opcode;
        begin
            opcode = desc[`CMD_OP_LSB +: `CMD_OP_W];
            while (!cmd_ready) @(negedge clk);
            assert (!busy)
            else $fatal(1, "[HOST] cmd_ready/busy mismatch before %s", label);

            current_cmd_is_gemm = (opcode == `CMD_OP_GEMM);
            current_cmd_acc_init = desc[`CMD_ACC_INIT_BIT];
            current_cmd_acc_clr = 0;
            if (opcode == `CMD_OP_GEMM) begin
                if (desc[`CMD_ACC_INIT_BIT]) acc_init_cmds++;
                else                         acc_noninit_cmds++;
            end

            cmd_desc  = desc;
            cmd_valid = 1;
            @(negedge clk);
            cmd_valid = 0;

            while (!done) @(negedge clk);
            while (!cmd_ready) @(negedge clk);
            assert (!busy)
            else $fatal(1, "[HOST] TPU stayed busy after %s", label);

            if (opcode == `CMD_OP_GEMM) begin
                if (desc[`CMD_ACC_INIT_BIT]) begin
                    if (current_cmd_acc_clr != 1) begin
                        $display("  [ERROR] %s ACC_INIT GEMM acc_clr count = %0d, expected 1",
                            label, current_cmd_acc_clr);
                        errors++;
                    end
                end else if (current_cmd_acc_clr != 0) begin
                    $display("  [ERROR] %s non-INIT GEMM acc_clr count = %0d, expected 0",
                        label, current_cmd_acc_clr);
                    errors++;
                end
            end
            current_cmd_is_gemm = 0;
        end
    endtask

    //=============================================================
    // ------------------------- Reset ----------------------------
    //=============================================================
    task automatic reset_dut();
        begin
            force clk = 0;
            rst_n     = 1;
            cmd_valid = 0;
            cmd_desc  = '0;
            // Streams run at full throughput.
            w_valid   = 1;
            a_valid   = 1;
            r_ready   = 1;
            #20 rst_n = 0;
            #20 rst_n = 1;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // --------------------- Command-Driven Run -------------------
    //=============================================================
    task automatic run_case(
        input int    M,
        input int    K,
        input int    N,
        input bit    fin,
        input bit    fout,
        input string nm
    );
        int MT, KT, NT;
        int exp_w, exp_a, exp_r, exp_ub, exp_ob, exp_acc_init, exp_acc_noninit;
        logic [`CMD_DESC_W-1:0] desc;
        begin
            MT = M / `ARRAY_S;
            KT = K / `ARRAY_S;
            NT = N / `ARRAY_S;
            exp_w  = (KT == 1) ? NT * `ARRAY_S : MT * NT * KT * `ARRAY_S;
            exp_a  = fin ? 0 : MT * NT * KT * `ARRAY_S;
            exp_r  = MT * NT * `ARRAY_S;
            exp_ub = fin ? 0 : MT * NT * KT * `ARRAY_S;
            exp_ob = fin ? MT * NT * KT * `ARRAY_S : (fout ? MT * NT * `ARRAY_S : 0);
            exp_acc_init    = MT * NT;
            exp_acc_noninit = MT * NT * (KT - 1);
            clear_sb();

            if (KT == 1) begin
                // Reuse the preloaded weight tile.
                for (int nt = 0; nt < NT; nt++) begin
                    desc = build_cmd_desc(`CMD_OP_LOAD_W, 0, 0, fin, fout, 0, 0, nt);
                    issue_cmd(desc, "LOAD_W");

                    desc = build_cmd_desc(`CMD_OP_PRELOAD_W, 0, 0, fin, fout, 0, 0, nt);
                    issue_cmd(desc, "PRELOAD_W");

                    for (int mt = 0; mt < MT; mt++) begin
                        if (!fin) begin
                            desc = build_cmd_desc(`CMD_OP_LOAD_A, 0, 0, fin, fout, mt, 0, nt);
                            issue_cmd(desc, "LOAD_A");
                        end

                        desc = build_cmd_desc(`CMD_OP_GEMM, 1, 1, fin, fout, mt, 0, nt);
                        issue_cmd(desc, "GEMM");

                        desc = build_cmd_desc(`CMD_OP_STORE_C, 0, 0, fin, fout, mt, 0, nt);
                        issue_cmd(desc, "STORE_C");
                    end
                end
            end else begin
                for (int mt = 0; mt < MT; mt++) begin
                    for (int nt = 0; nt < NT; nt++) begin
                        for (int kt = 0; kt < KT; kt++) begin
                            desc = build_cmd_desc(`CMD_OP_LOAD_W, 0, 0, fin, fout, mt, kt, nt);
                            issue_cmd(desc, "LOAD_W");

                            desc = build_cmd_desc(`CMD_OP_PRELOAD_W, 0, 0, fin, fout, mt, kt, nt);
                            issue_cmd(desc, "PRELOAD_W");

                            if (!fin) begin
                                desc = build_cmd_desc(`CMD_OP_LOAD_A, 0, 0, fin, fout, mt, kt, nt);
                                issue_cmd(desc, "LOAD_A");
                            end

                            desc = build_cmd_desc(`CMD_OP_GEMM, (kt == 0), (kt == KT - 1),
                                fin, fout, mt, kt, nt);
                            issue_cmd(desc, "GEMM");
                        end

                        desc = build_cmd_desc(`CMD_OP_STORE_C, 0, 0, fin, fout, mt, KT - 1, nt);
                        issue_cmd(desc, "STORE_C");
                    end
                end
            end

            $display("  %-20s w=%0d/%0d a=%0d/%0d r=%0d/%0d ub=%0d/%0d acc_clr=%0d/%0d ob_wr=%0d ob_rd=%0d",
                nm, w_beats, exp_w, a_beats, exp_a, r_beats, exp_r,
                ub_rd, exp_ub, acc_clr_total, exp_acc_init, ob_wr, ob_rd);

            if (w_beats != exp_w)  begin $display("  [ERROR] %s weight beats", nm); errors++; end
            if (a_beats != exp_a)  begin $display("  [ERROR] %s activation beats", nm); errors++; end
            if (r_beats != exp_r)  begin $display("  [ERROR] %s result beats", nm); errors++; end
            if (ub_wr   != exp_ub) begin $display("  [ERROR] %s UB writes", nm); errors++; end
            if (ub_rd   != exp_ub) begin $display("  [ERROR] %s UB reads", nm); errors++; end
            if (acc_clr_total != exp_acc_init) begin
                $display("  [ERROR] %s acc_clr total", nm);
                errors++;
            end
            if (acc_init_cmds != exp_acc_init || acc_noninit_cmds != exp_acc_noninit) begin
                $display("  [ERROR] %s ACC_INIT descriptor coverage", nm);
                errors++;
            end

            if (fout) begin
                if (ob_wr != MT * NT * `ARRAY_S) begin
                    $display("  [ERROR] %s OBUF writes", nm);
                    errors++;
                end
                // Fused producer
                for (int i = 0; i < ob_wr; i++) begin
                    if (ob_wr_seq[i] != i) begin
                        $display("  [ERROR] %s OBUF wr addr[%0d]=%0d expected %0d",
                            nm, i, ob_wr_seq[i], i);
                        errors++;
                    end
                end
            end else if (ob_wr != 0) begin
                $display("  [ERROR] %s wrote OBUF without fuse_out", nm);
                errors++;
            end

            if (fin) begin
                if (ob_rd != exp_ob) begin
                    $display("  [ERROR] %s OBUF reads", nm);
                    errors++;
                end
                // Fused consumer
                for (int i = 0; i < ob_rd; i++) begin
                    if (ob_rd_seq[i] != i) begin
                        $display("  [ERROR] %s OBUF rd addr[%0d]=%0d expected %0d",
                            nm, i, ob_rd_seq[i], i);
                        errors++;
                    end
                end
            end else if (ob_rd != 0) begin
                $display("  [ERROR] %s read OBUF without fuse_in", nm);
                errors++;
            end
        end
    endtask

    task automatic run_nop();
        logic [`CMD_DESC_W-1:0] desc;
        begin
            clear_sb();
            desc = build_cmd_desc(`CMD_OP_NOP, 0, 0, 0, 0, 0, 0, 0);
            issue_cmd(desc, "NOP");
            if (w_beats != 0 || a_beats != 0 || r_beats != 0 ||
                ub_wr != 0 || ub_rd != 0 || ob_wr != 0 || ob_rd != 0) begin
                $display("  [ERROR] NOP caused a datapath transfer");
                errors++;
            end
            $display("  NOP                  no datapath transfer");
        end
    endtask

    //=============================================================
    // -------------------------- SVA -----------------------------
    //=============================================================
    `ifdef SVA
        CHECK_IDLE_BUSY: assert property (@(posedge clk) disable iff(!rst_n)
            !(cmd_ready && busy));
        CHECK_SERIAL_HOST: assert property (@(posedge clk) disable iff(!rst_n)
            cmd_valid |-> (cmd_ready && !busy));
        CHECK_CMD_PULSE: assert property (@(posedge clk) disable iff(!rst_n)
            cmd_valid |=> !cmd_valid);
        CHECK_CMD_OPCODE: assert property (@(posedge clk) disable iff(!rst_n)
            (cmd_valid && cmd_ready) |->
            (cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_NOP ||
             cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_LOAD_W ||
             cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_PRELOAD_W ||
             cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_LOAD_A ||
             cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_GEMM ||
             cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] == `CMD_OP_STORE_C));
        CHECK_ACC_FLAGS: assert property (@(posedge clk) disable iff(!rst_n)
            (cmd_valid && cmd_ready && cmd_desc[`CMD_OP_LSB +: `CMD_OP_W] != `CMD_OP_GEMM) |->
            (!cmd_desc[`CMD_ACC_INIT_BIT] && !cmd_desc[`CMD_ACC_FINAL_BIT]));
    `endif

    //=============================================================
    // ------------------------- Watchdog -------------------------
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
        $display("  TSC SERIAL COMMAND CONTROL CHECK");
        $display("======================================");
        reset_dut();
        run_nop();
        run_case(  16,   16,   16, 0, 0, "16x16x16");
        run_case(  32,   48,   32, 0, 0, "32x48x32");
        run_case(  64,   16,   32, 0, 0, "64x16x32");
        run_case(  16, 1024,   16, 0, 0, "16x1024x16 bigK");
        run_case(  16, 4096,   16, 0, 0, "16x4096x16 maxK");
        run_case( 128,   16,   16, 0, 0, "128x16x16 bigM");
        run_case(  16,   16,  256, 0, 0, "16x16x256 bigN");
        run_case(  16,   16,   32, 0, 1, "fuse_src");
        run_case(  16,   32,   16, 1, 0, "fuse_dst K=32");
        run_case(  16,   16,  128, 0, 1, "fuse_src_128");
        run_case(  16,  128,   16, 1, 0, "fuse_dst K=128");
        $display("======================================");
        if (errors == 0) $display("  [SUCCESS] TSC CONTROL CHECK PASS");
        else             $display("  [FAILURE] %0d ERRORS", errors);
        $display("======================================");
        $finish;
    end

endmodule
