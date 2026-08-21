/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

`define CLK_PERIOD 1
`define TIMEOUT 1000000
`define TEST_NUM 500

module PATTERN(
    output logic clk,
    output logic [`AMEM_DW-1:0] data_i,
    output logic [`AMEM_ADDR_W-1:0] addr_w,
    output logic [`AMEM_ADDR_W-1:0] addr_r,
    output logic en_w,
    output logic en_r,
    input [`AMEM_DW-1:0] data_o
);

    //=============================================================
    // ------- Parameters, Integers & Reference Model -------------
    //=============================================================
    typedef enum int {PHASE1, PHASE2, PHASE3, PHASE4, PHASE5, DONE} phase_type;
    phase_type veri_phase;
    reg [`AMEM_DW-1:0] gold [0:`AMEM_D-1];

    //=============================================================
    // ------------------- Clock & Reset --------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    task automatic init_dut();
        begin
            force clk = 0;
            data_i = 0;
            addr_w = 0;
            addr_r = 0;
            en_w = 0;
            en_r = 0;
            #20;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // --------------------- Data Generator -----------------------
    //=============================================================
    function automatic [`AMEM_DW-1:0] rand_data();
        begin
            rand_data = {$random, $random, $random, $random}; // 128b -> truncated to AMEM_DW (112)
        end
    endfunction

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================

    // Write
    task automatic mem_write(input [`AMEM_ADDR_W-1:0] a, input [`AMEM_DW-1:0] d);
        begin
            addr_w = a;
            data_i = d;
            en_w = 1;
            en_r = 0;
            @(negedge clk);
            gold[a] = d; // save into gold
        end
    endtask

    // Read
    task automatic mem_read_check(input [`AMEM_ADDR_W-1:0] a);
        begin
            addr_r = a;
            en_r = 1;
            en_w = 0;
            @(negedge clk);
            CHECK_READ: assert (data_o === gold[a])
            else $fatal(1, "[ERROR] : READ addr=%0d exp=%028h got=%028h", a, gold[a], data_o);
        end
    endtask

    // Check write disable.
    task automatic en_w_disable_check(input [`AMEM_ADDR_W-1:0] a);
        logic [`AMEM_DW-1:0] known;
        begin
            // Write known data.
            known = rand_data();
            mem_write(a, known);

            // Attempt disabled write.
            addr_w = a;
            data_i = ~known;
            en_w = 0;
            en_r = 0;
            @(negedge clk);

            // Check preserved data.
            addr_r = a;
            en_r = 1;
            en_w = 0;
            @(negedge clk);
            CHECK_EN_W: assert (data_o === gold[a])
            else $fatal(1, "[ERROR] : write NOT inhibited when en_w = 0.");
        end
    endtask

    // Check dual-port concurrency.
    task automatic dual_port_stream();
        logic [`AMEM_DW-1:0] exp_r;
        int aw, ar;
        logic [`AMEM_DW-1:0] d;
        bit we;
        begin
            en_r = 1;
            for (int k = 0; k < `TEST_NUM; k++) begin
                // Check previous read.
                if (k > 0) begin
                    CHECK_DUAL: assert (data_o === exp_r)
                    else $fatal(1, "[ERROR] : dual-port read exp=%028h got=%028h", exp_r, data_o);
                end

                // Drive current read/write.
                aw = $urandom_range(0, `AMEM_D-1);
                ar = $urandom_range(0, `AMEM_D-1);
                d = rand_data();
                we = $urandom_range(0, 1);

                // Avoid same-address read/write.
                if (we && ar == aw) ar = (aw + 1) % `AMEM_D;

                addr_w = aw[`AMEM_ADDR_W-1:0];
                addr_r = ar[`AMEM_ADDR_W-1:0];
                data_i = d;
                en_w = we;

                // Read a different address.
                exp_r = gold[ar];
                if (we) gold[aw] = d;
                @(negedge clk);
            end

            // Check final read.
            CHECK_DUAL_END: assert (data_o === exp_r)
            else $fatal(1, "[ERROR] : dual-port read exp=%028h got=%028h", exp_r, data_o);
            en_w = 0;
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
        init_dut();
        // Write/read sweep
        veri_phase = PHASE1;
        for (int t = 0; t < `TEST_NUM; t++) begin
            for (int a = 0; a < `AMEM_D; a++) mem_write(a[`AMEM_ADDR_W-1:0], rand_data());
            for (int a = 0; a < `AMEM_D; a++) mem_read_check(a[`AMEM_ADDR_W-1:0]);
        end
        $display("Phase 1 Pass.");

        // Random read/write
        veri_phase = PHASE2;
        for (int t = 0; t < `TEST_NUM; t++) begin
            int a = $urandom_range(0, `AMEM_D-1);
            if ($urandom_range(0, 1)) mem_write(a[`AMEM_ADDR_W-1:0], rand_data());
            else mem_read_check(a[`AMEM_ADDR_W-1:0]);
        end
        $display("Phase 2 Pass.");

        // Write/read turnaround
        veri_phase = PHASE3;
        for (int t = 0; t < `TEST_NUM; t++) begin
            for (int a = 0; a < `AMEM_D; a++) begin
                mem_write(a[`AMEM_ADDR_W-1:0], rand_data());
                mem_read_check(a[`AMEM_ADDR_W-1:0]);
            end
        end
        $display("Phase 3 Pass.");

        // Write disable
        veri_phase = PHASE4;
        for (int t = 0; t < `TEST_NUM; t++)
            for (int a = 0; a < `AMEM_D; a++) en_w_disable_check(a[`AMEM_ADDR_W-1:0]);
        $display("Phase 4 Pass.");

        // Dual-port concurrency
        veri_phase = PHASE5;
        dual_port_stream();
        $display("Phase 5 Pass.");

        veri_phase = DONE;
        #100 end_task();
        $finish;
    end

endmodule
