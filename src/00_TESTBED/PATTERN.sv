/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    PATTERN.sv
* Project:      tpu-accelerator-lite
* Module:       PATTERN
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/
`include "define.vh"

module PATTERN (
    output logic clk,
    output logic rst_n,
    output logic [`HOST_DATA_W-1:0] host_data_i,
    output logic [`HOST_TYPE_W-1:0] host_type,
    output logic host_valid,
    input logic host_ready,
    input logic [`HOST_DATA_W-1:0] host_data_o,
    input logic host_out_valid,
    output logic host_out_ready,
    input logic busy,
    input logic done
);

    localparam CLK_PERIOD = 10;
    localparam CMD_TIMEOUT = 100000;
    `ifdef TIMEOUT
        localparam SIM_TIMEOUT = `TIMEOUT;
    `else
        localparam SIM_TIMEOUT = 1000000000;
    `endif

    //=============================================================
    //                         Test Suite
    //=============================================================
    `include "pattern/test_suite.svh"

    //=============================================================
    //                        Pattern Data
    //=============================================================
    logic [`CMD_DESC_W-1:0] mem_cmd [0:MAX_CMD_COUNT-1];
    logic [`W_RAW_BW-1:0] mem_w [0:MAX_W_ROWS-1];
    logic [`A_BW-1:0] mem_a [0:MAX_A_ROWS-1];
    logic [`R_BW-1:0] mem_g [0:MAX_R_ROWS-1];

    int cur_test;
    int cmd_ptr;
    int w_ptr;
    int a_ptr;
    int r_ptr;
    int err_this;
    int tests_run;
    int tests_pass;
    int tests_fail;
    int selected_test;
    longint cyc;
    longint cyc_mark;

    //=============================================================
    //                            Clock
    //=============================================================
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    initial begin : CYCLE_COUNTER
        cyc = 0;
        forever @(posedge clk) cyc <= cyc + 1;
    end

    //=============================================================
    //                            Reset
    //=============================================================
    task automatic drive_reset();
        host_data_i = '0;
        host_type = `HOST_TYPE_CMD;
        host_valid = 1'b0;
        host_out_ready = 1'b0;

        rst_n = 1'b1;
        force clk = 1'b0;
        #20 rst_n = 1'b0;
        #20 rst_n = 1'b1;
        release clk;
        @(negedge clk);
    endtask

    //=============================================================
    //                      Descriptor Check
    //=============================================================
    task automatic check_descriptor(
        input logic [`CMD_DESC_W-1:0] desc,
        input int index
    );
        logic [`CMD_OP_W-1:0] opcode;
        logic [`K_VALID_W-1:0] k_valid;
        logic [`ACT_MODE_W-1:0] act_mode;
        opcode = desc[`CMD_OP_LSB +: `CMD_OP_W];
        k_valid = desc[`CMD_K_VALID_LSB +: `K_VALID_W];
        act_mode = desc[`CMD_ACT_MODE_LSB +: `ACT_MODE_W];

        if($isunknown(desc)) $fatal(1, "[ERROR] : t%02d cmd%0d contains X/Z", cur_test, index);
        if(desc[7:5] != 0 || desc[`CMD_DESC_W-1:`CMD_DEFINED_W] != 0) $fatal(1, "[ERROR] : t%02d cmd%0d sets reserved bits", cur_test, index);
        if(opcode > `CMD_OP_LOAD_BIAS) $fatal(1, "[ERROR] : t%02d cmd%0d has invalid opcode", cur_test, index);
        if(opcode != `CMD_OP_GEMM &&
            (desc[`CMD_ACC_INIT_BIT] || desc[`CMD_ACC_FINAL_BIT])) $fatal(1, "[ERROR] : t%02d cmd%0d has invalid ACC flags", cur_test, index);
        if(opcode == `CMD_OP_GEMM &&
            (k_valid == 0 || k_valid > `ARRAY_S)) $fatal(1, "[ERROR] : t%02d cmd%0d has invalid K_VALID", cur_test, index);
        if(opcode != `CMD_OP_GEMM && k_valid != 0) $fatal(1, "[ERROR] : t%02d cmd%0d sets K_VALID", cur_test, index);
        if(desc[`CMD_BIAS_EN_BIT] &&
            (opcode != `CMD_OP_GEMM || !desc[`CMD_ACC_INIT_BIT])) $fatal(1, "[ERROR] : t%02d cmd%0d has invalid BIAS_EN", cur_test, index);
        if(opcode == `CMD_OP_STORE_C && act_mode > `ACT_RELU) $fatal(1, "[ERROR] : t%02d cmd%0d has invalid ACT_MODE", cur_test, index);
        if(opcode != `CMD_OP_STORE_C && act_mode != `ACT_NONE) $fatal(1, "[ERROR] : t%02d cmd%0d sets ACT_MODE", cur_test, index);
        if(opcode != `CMD_OP_LOAD_A && desc[`CMD_A_MASK_EN_BIT]) $fatal(1, "[ERROR] : t%02d cmd%0d sets A_MASK_EN", cur_test, index);
    endtask

    //=============================================================
    //                          File Load
    //=============================================================
    task automatic load_case_files(
        input int test_index
    );
        string cmd_file;
        string w_file;
        string a_file;
        string g_file;
        cmd_file = $sformatf("../00_TESTBED/pattern/t%02d_command.dat", test_index);
        w_file = $sformatf("../00_TESTBED/pattern/t%02d_weight.dat", test_index);
        a_file = $sformatf("../00_TESTBED/pattern/t%02d_activation.dat", test_index);
        g_file = $sformatf("../00_TESTBED/pattern/t%02d_golden.dat", test_index);

        for(int i = 0; i < TEST_CMD_COUNT[test_index]; i++) mem_cmd[i] = 'x;
        for(int i = 0; i < TEST_W_ROWS[test_index]; i++) mem_w[i] = 'x;
        for(int i = 0; i < TEST_A_ROWS[test_index]; i++) mem_a[i] = 'x;
        for(int i = 0; i < TEST_R_ROWS[test_index]; i++) mem_g[i] = 'x;

        $readmemh(cmd_file, mem_cmd, 0, TEST_CMD_COUNT[test_index]-1);
        if(TEST_W_ROWS[test_index] > 0) $readmemh(w_file, mem_w, 0, TEST_W_ROWS[test_index]-1);
        if(TEST_A_ROWS[test_index] > 0) $readmemh(a_file, mem_a, 0, TEST_A_ROWS[test_index]-1);
        if(TEST_R_ROWS[test_index] > 0) $readmemh(g_file, mem_g, 0, TEST_R_ROWS[test_index]-1);

        for(int i = 0; i < TEST_CMD_COUNT[test_index]; i++) if($isunknown(mem_cmd[i])) $fatal(1, "[ERROR] : failed to load %s", cmd_file);
        for(int i = 0; i < TEST_W_ROWS[test_index]; i++) if($isunknown(mem_w[i])) $fatal(1, "[ERROR] : failed to load %s", w_file);
        for(int i = 0; i < TEST_A_ROWS[test_index]; i++) if($isunknown(mem_a[i])) $fatal(1, "[ERROR] : failed to load %s", a_file);
        for(int i = 0; i < TEST_R_ROWS[test_index]; i++) if($isunknown(mem_g[i])) $fatal(1, "[ERROR] : failed to load %s", g_file);
    endtask

    //=============================================================
    //                      Host Input Tasks
    //=============================================================
    task automatic send_host_packet(
        input logic [`HOST_TYPE_W-1:0] packet_type,
        input logic [`CMD_DESC_W-1:0] packet_data,
        input int beat_count
    );
        for(int beat = 0; beat < beat_count; beat++) begin
            host_type = packet_type;
            host_data_i = packet_data[beat*`HOST_DATA_W +: `HOST_DATA_W];
            host_valid = 1'b1;
            while(!host_ready) @(negedge clk);
            @(negedge clk);
        end
        host_data_i = '0;
        host_type = `HOST_TYPE_CMD;
        host_valid = 1'b0;
    endtask

    task automatic issue_command(
        input logic [`CMD_DESC_W-1:0] desc,
        input int index
    );
        check_descriptor(desc, index);
        if(busy) $fatal(1, "[ERROR] : TPU busy before t%02d cmd%0d", cur_test, index);
        send_host_packet(`HOST_TYPE_CMD, desc, `HOST_CMD_BEATS);
    endtask

    task automatic wait_command_done(
        input int index
    );
        int wait_cycles;
        wait_cycles = 0;
        while(!done && wait_cycles < CMD_TIMEOUT) begin
            wait_cycles++;
            @(negedge clk);
        end
        if(!done) $fatal(1, "[ERROR] : t%02d cmd%0d timeout", cur_test, index);

        while(busy) @(negedge clk);
        if(busy) $fatal(1, "[ERROR] : TPU remains busy after t%02d cmd%0d", cur_test, index);
    endtask

    //=============================================================
    //                        Stream Tasks
    //=============================================================
    task automatic feed_weight_stream(
        input int beat_count
    );
        int gap_cycles;
        logic [`CMD_DESC_W-1:0] packet_data;
        for(int row = 0; row < beat_count; row++) begin
            if(w_ptr >= TEST_W_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d weight stream overflow", cur_test);

            case(TEST_WSTALL[cur_test])
                1: gap_cycles = (row % 4 == 1)? 1 : 0;
                2: gap_cycles = (row % 5 == 2)? 3 : 0;
                default: gap_cycles = 0;
            endcase

            host_valid = 1'b0;
            repeat(gap_cycles) @(negedge clk);
            packet_data = '0;
            packet_data[`W_RAW_BW-1:0] = mem_w[w_ptr];
            send_host_packet(`HOST_TYPE_WEIGHT, packet_data, `HOST_WEIGHT_BEATS);
            w_ptr++;
        end
    endtask

    task automatic feed_activation_stream(
        input int beat_count
    );
        int gap_cycles;
        logic [`CMD_DESC_W-1:0] packet_data;
        for(int row = 0; row < beat_count; row++) begin
            if(a_ptr >= TEST_A_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d activation stream overflow", cur_test);

            case(TEST_ASTALL[cur_test])
                1: gap_cycles = (row % 4 == 1)? 1 : 0;
                2: gap_cycles = (row % 5 == 2)? 3 : 0;
                default: gap_cycles = 0;
            endcase

            host_valid = 1'b0;
            repeat(gap_cycles) @(negedge clk);
            packet_data = '0;
            packet_data[`A_BW-1:0] = mem_a[a_ptr];
            send_host_packet(`HOST_TYPE_ACTIVATION, packet_data, `HOST_ACTIVATION_BEATS);
            a_ptr++;
        end
    endtask

    task automatic monitor_result_tile();
        int word;
        int beat;
        int phase;
        logic hold_active;
        logic [`HOST_DATA_W-1:0] hold_data;
        logic [`R_BW-1:0] result_word;
        word = 0;
        beat = 0;
        phase = 0;
        hold_active = 0;
        hold_data = '0;
        while(word < `ARRAY_S) begin
            result_word = '0;
            beat = 0;
            while(beat < `HOST_RESULT_BEATS) begin
                case(TEST_RSTALL[cur_test])
                    1: host_out_ready = (phase % 5 != 2);
                    2: host_out_ready = !((phase % 9 >= 2) && (phase % 9 <= 5));
                    default: host_out_ready = 1'b1;
                endcase

                if($isunknown(host_out_valid)) $fatal(1, "[ERROR] : t%02d host_out_valid is X/Z", cur_test);
                if(host_out_valid && $isunknown(host_data_o)) $fatal(1, "[ERROR] : t%02d result beat contains X/Z", cur_test);

                if(hold_active) begin
                    if(!host_out_valid) $fatal(1, "[ERROR] : t%02d host_out_valid dropped during stall", cur_test);
                    if(host_data_o !== hold_data) $fatal(1, "[ERROR] : t%02d host_data_o changed during stall", cur_test);
                end

                if(host_out_valid && !host_out_ready && !hold_active) begin
                    hold_active = 1'b1;
                    hold_data = host_data_o;
                end

                if(host_out_valid && host_out_ready) begin
                    result_word[beat*`HOST_DATA_W +: `HOST_DATA_W] = host_data_o;
                    hold_active = 1'b0;
                    beat++;
                end
                phase++;
                if(phase > CMD_TIMEOUT) $fatal(1, "[ERROR] : t%02d result timeout", cur_test);
                @(negedge clk);
            end

            if(r_ptr >= TEST_R_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d result stream overflow", cur_test);
            if(result_word !== mem_g[r_ptr]) begin
                if(err_this < 8) $display("  [MISMATCH] t%02d beat=%0d exp=%064h got=%064h",
                    cur_test, r_ptr, mem_g[r_ptr], result_word);
                err_this++;
            end
            word++;
            r_ptr++;
        end
        host_out_ready = 1'b0;
    endtask

    //=============================================================
    //                        Trace Replay
    //=============================================================
    task automatic replay_trace();
        logic [`CMD_DESC_W-1:0] desc;
        logic [`CMD_OP_W-1:0] opcode;
        for(cmd_ptr = 0; cmd_ptr < TEST_CMD_COUNT[cur_test]; cmd_ptr++) begin
            desc = mem_cmd[cmd_ptr];
            opcode = desc[`CMD_OP_LSB +: `CMD_OP_W];
            issue_command(desc, cmd_ptr);

            case(opcode)
                `CMD_OP_LOAD_W: feed_weight_stream(`ARRAY_S);
                `CMD_OP_LOAD_BIAS: feed_weight_stream(`BIAS_BEATS);
                `CMD_OP_LOAD_A: feed_activation_stream(
                    `ARRAY_S +
                    ((desc[`CMD_A_MASK_EN_BIT])? `AMASK_BEATS : 0)
                );
                `CMD_OP_STORE_C,
                `CMD_OP_READ_UB: monitor_result_tile();
                `CMD_OP_NOP,
                `CMD_OP_PRELOAD_W,
                `CMD_OP_GEMM: ;
                default: $fatal(1, "[ERROR] : unsupported opcode %0d", opcode);
            endcase
            wait_command_done(cmd_ptr);
        end

        if(w_ptr != TEST_W_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d consumed %0d/%0d weight rows",
            cur_test, w_ptr, TEST_W_ROWS[cur_test]);
        if(a_ptr != TEST_A_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d consumed %0d/%0d activation rows",
            cur_test, a_ptr, TEST_A_ROWS[cur_test]);
        if(r_ptr != TEST_R_ROWS[cur_test]) $fatal(1, "[ERROR] : t%02d consumed %0d/%0d result rows",
            cur_test, r_ptr, TEST_R_ROWS[cur_test]);
    endtask

    //=============================================================
    //                         Pass / Fail
    //=============================================================
    task automatic judge();
        if(err_this == 0) begin
            tests_pass++;
            $display("  [PASS] t%02d %-22s cyc=%0d",
                cur_test, TEST_NAME[cur_test], cyc - cyc_mark);
        end
        else begin
            tests_fail++;
            $display("  [FAIL] t%02d %-22s err=%0d cyc=%0d",
                cur_test, TEST_NAME[cur_test], err_this, cyc - cyc_mark);
        end
    endtask

    task automatic end_task();
        $display("============================================");
        if(tests_fail == 0) $display("  [SUCCESS] ALL %0d / %0d TESTS PASS !", tests_pass, tests_run);
        else $display("  [FAILURE] %0d / %0d TESTS FAILED", tests_fail, tests_run);
        $display("  total cycles : %0d", cyc);
        $display("============================================");
    endtask

    //=============================================================
    //                          Watchdog
    //=============================================================
    initial begin
        #(SIM_TIMEOUT);
        $fatal(1, "[ERROR] : Simulation timeout in t%02d", cur_test);
    end

    //=============================================================
    //                          Main Flow
    //=============================================================
    initial begin
        cyc_mark = 0;
        cur_test = 0;
        tests_run = 0;
        tests_pass = 0;
        tests_fail = 0;
        selected_test = -1;

        if($value$plusargs("TEST_ID=%d", selected_test)) begin
            if(selected_test < 0 || selected_test >= NUM_TESTS) $fatal(1, "[ERROR] : TEST_ID %0d is out of range", selected_test);
        end

        for(int test_index = 0; test_index < NUM_TESTS; test_index++) begin
            if(selected_test < 0 || selected_test == test_index) begin
                cur_test = test_index;
                tests_run++;
                drive_reset();
                load_case_files(test_index);
                cmd_ptr = 0;
                w_ptr = 0;
                a_ptr = 0;
                r_ptr = 0;
                err_this = 0;
                cyc_mark = cyc;

                $display("  [RUN] t%02d %-22s M=%0d K=%0d N=%0d",
                    test_index, TEST_NAME[test_index],
                    TEST_M[test_index], TEST_K[test_index], TEST_N[test_index]);
                replay_trace();
                judge();
            end
        end

        end_task();
        if(tests_fail != 0) $fatal(1, "[ERROR] : TPU verification failed");
        $finish;
    end

    //=============================================================
    //                 SystemVerilog Assertion
    //=============================================================
    HOST_INPUT_STABLE: assert property(
        @(posedge clk) disable iff(!rst_n)
        host_valid && !host_ready |=> host_valid && $stable({host_type, host_data_i})
    )
    else $fatal(1, "[ERROR]: Host input changed while stalled");

    HOST_OUTPUT_STABLE: assert property(
        @(posedge clk) disable iff(!rst_n)
        host_out_valid && !host_out_ready |=> host_out_valid && $stable(host_data_o)
    )
    else $fatal(1, "[ERROR]: Host output changed while stalled");

    CHECK_RESET_VALID_LOW: assert property(
        @(posedge clk)
        !rst_n |-> (!host_out_valid && !busy && !done)
    )
    else $fatal(1, "[ERROR]: Output valid or status asserted during reset");

endmodule
