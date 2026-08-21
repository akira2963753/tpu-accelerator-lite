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
    output logic [`WMEM_ADDR_W-1:0] addr_w,
    output logic [`WMEM_DW-1:0] data_i,
    output logic en_c,
    output logic en_w,
    input [`WMEM_DW-1:0] data_o
);

    //=============================================================
    // ------- Parameters, Integers & Reference Model -------------
    //=============================================================
    typedef enum int {PHASE1, PHASE2, PHASE3, PHASE4, DONE} phase_type;
    phase_type veri_phase;
    reg [`WMEM_DW-1:0] gold [0:`WMEM_D-1];

    //=============================================================
    // ------------------- Clock & Reset --------------------------
    //=============================================================
    always #(`CLK_PERIOD/2.0) clk = ~clk;

    task automatic init_dut();
        int i;
        begin
            force clk = 0;
            addr_w = 0;
            data_i = 0;
            en_c = 0;
            en_w = 0;
            #20;
            release clk;
            @(negedge clk);
        end
    endtask

    //=============================================================
    // --------------------- Data Generator -----------------------
    //=============================================================
    function automatic [`WMEM_DW-1:0] rand_data();
        begin
            rand_data = {$random, $random, $random}; // 96b -> truncated to WMEM_DW (80)
        end
    endfunction

    //=============================================================
    // ---------------------------- Tasks -------------------------
    //=============================================================

    // Write
    task automatic mem_write(input [`WMEM_ADDR_W-1:0] a, input [`WMEM_DW-1:0] d);
        begin
            addr_w = a;
            data_i = d;
            en_c = 1;
            en_w = 1;
            @(negedge clk);
            gold[a] = d; // save into gold
        end
    endtask

    // Read
    task automatic mem_read_check(input [`WMEM_ADDR_W-1:0] a);
        begin
            addr_w = a;
            en_c = 1;
            en_w = 0;
            @(negedge clk);
            CHECK_READ: assert (data_o === gold[a])
            else $fatal(1, "[ERROR] : READ addr=%0d exp=%020h got=%020h", a, gold[a], data_o);
        end
    endtask

    // Check write disable.
    task automatic en_c_disable_check(input [`WMEM_ADDR_W-1:0] a);
        logic [`WMEM_DW-1:0] known;
        begin
            // Write known data.
            known = rand_data();
            mem_write(a, known);        

            // Attempt disabled write.
            addr_w = a;
            data_i = ~known;
            en_c = 0;
            en_w = 1;
            @(negedge clk);

            // Check preserved data.
            addr_w = a;
            en_c = 1;
            en_w = 0;
            @(negedge clk);
            CHECK_EN_C: assert (data_o === gold[a])
            else $fatal(1, "[ERROR] : write would be stop when en_c = 0.");
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
        // Write sweep
        veri_phase = PHASE1;
        for(int t = 0; t < `TEST_NUM; t++) begin
            for (int a = 0; a < `WMEM_D; a++) mem_write(a[`WMEM_ADDR_W-1:0], rand_data());
            for (int a = 0; a < `WMEM_D; a++) mem_read_check(a[`WMEM_ADDR_W-1:0]);
        end
        $display("Phase 1 Pass.");

        // Random read/write
        veri_phase = PHASE2;
        for (int t = 0; t < `TEST_NUM; t++) begin
            int a = $urandom_range(0, `WMEM_D-1);
            if ($urandom_range(0, 1)) mem_write(a[`WMEM_ADDR_W-1:0], rand_data());
            else mem_read_check(a[`WMEM_ADDR_W-1:0]);
        end
        $display("Phase 2 Pass.");

        // Write/read turnaround
        veri_phase = PHASE3;
        for (int t = 0; t < `TEST_NUM; t++) begin
            for (int a = 0; a < `WMEM_D; a++) begin
                mem_write(a[`WMEM_ADDR_W-1:0], rand_data());
                mem_read_check(a[`WMEM_ADDR_W-1:0]);
            end
        end
        $display("Phase 3 Pass.");

        // Write disable
        veri_phase = PHASE4;
        for (int t = 0; t < `TEST_NUM; t++)
            for (int a = 0; a < `WMEM_D; a++) en_c_disable_check(a[`WMEM_ADDR_W-1:0]);
        $display("Phase 4 Pass.");

        veri_phase = DONE;
        #100 end_task();
        $finish;
    end

endmodule
