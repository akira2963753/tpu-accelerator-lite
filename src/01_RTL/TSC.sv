/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TSC.sv
* Project:      tpu-accelerator-lite
* Module:       TSC (Tensor Processing Unit System Controller)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module TSC (
    input clk,
    input rst_n,

    // external : job descriptor + status (valid/ready handshake)
    input cmd_valid,
    output logic cmd_ready,
    input [`M_W-1:0] dim_m, // M/K/N assumed multiples of ARRAY_S
    input [`K_W-1:0] dim_k,
    input [`N_W-1:0] dim_n,
    input [`QMULT_W-1:0] quant_mult, // per-tensor requant multiplier M0 (latched with dims)
    input [`QSHIFT_W-1:0] quant_shift, // per-tensor requant right-shift n (latched with dims)
    input fuse_in, // 1 = take activation from OBUF (previous layer's output), no host stream
    input fuse_out, // 1 = also write this job's results into OBUF for the next layer
    output logic busy,
    output logic done,

    // external : data load / result streaming (valid/ready)
    input w_valid, // raw weight beat available
    output logic w_ready,
    input a_valid, // activation beat available
    output logic a_ready,
    output logic r_valid, // result beat available
    input r_ready,

    // internal : weight write path (WPU -> WMEM)
    output logic wpu_wmem_w, // WPU encode strobe (load)
    output logic [`WMEM_ADDR_W-1:0] wmem_addr, // shared : load write (via WPU) / preload read
    output logic wmem_en_c, // WMEM chip enable
    output logic wmem_en_w, // WMEM write enable (1=write, 0=read)

    // internal : UB activation window (dual-port)
    output logic [`AMEM_ADDR_W-1:0] ub_addr_w,
    output logic [`AMEM_ADDR_W-1:0] ub_addr_r,
    output logic ub_en_w,
    output logic ub_en_r,

    // internal : OBUF layer-fusion buffer (dual-port)
    output logic [`OBUF_ADDR_W-1:0] obuf_addr_w,
    output logic [`OBUF_ADDR_W-1:0] obuf_addr_r,
    output logic obuf_en_w,
    output logic obuf_en_r,
    output logic act_sel, // 1 = Data_Setup takes OBUF data instead of UB data

    // internal : systolic arrays / skew
    output logic weight_valid, // 1=PRELOAD push weights, 0=CAL compute
    output logic skew_en, // Data_Setup skew advance
    output logic comp_clr, // clear Data_Setup compensation state at each tile load
    output logic csa_cw_valid, // CSA compensation-weight valid
    output logic csa_act_valid, // CSA compensation-activation valid

    // internal : accumulator (ACC) / output
    output logic acc_clr, // clear ACC block at (mt,nt) start
    output logic [`ARRAY_S-1:0] acc_wr_en, // per-column psum accumulate (de-skew staggered)
    output logic [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row, // per-column output-row address
    output logic cacc_en, // compensation add window (every k-tile ; ACC gates by cout_valid)
    output logic [`ROW_IDX_W-1:0] cacc_wr_row, // compensation output-row address
    output logic acc_rd_en, // read one ACC row out (OUT)
    output logic [`ROW_IDX_W-1:0] acc_rd_row,

    // internal : requant constants held for OP (latched from the descriptor)
    output logic [`QMULT_W-1:0] quant_mult_q,
    output logic [`QSHIFT_W-1:0] quant_shift_q
);

    localparam ACT_N = `ARRAY_S; // 16 activation vectors / tile
    localparam RESULT_LAT = `ARRAY_S + 1; // RSA col-0 psum latency (pinned by sim : +1 for de-skew chain reg)
    localparam CACC_LAT = 2; // CSA compensation latency (column-aligned, no skew)
    localparam CAL_DRAIN = 2*`ARRAY_S + 4; // must cover RESULT_LAT + column de-skew (15) tail
    localparam CAL_TOTAL = ACT_N + CAL_DRAIN;
    localparam OBUF_TSEL_W = `OBUF_ADDR_W - `ROW_IDX_W; // OBUF 16-row tile select = 3

    typedef enum logic [2:0] {IDLE, LOADING, PRELOAD_W, CAL, OUT, DONE} STATETYPE;
    STATETYPE state, nx_state;

    // C [M x N] = A [M x K] x W [K x N]
    logic [`M_W-1:0] m_tiles;
    logic [`K_W-1:0] k_tiles;
    logic [`N_W-1:0] n_tiles;

    // tile counter
    logic [`M_W-1:0] mt;
    logic [`K_W-1:0] kt;
    logic [`N_W-1:0] nt;

    // phase counters
    logic [4:0] pw; // preload_w counter
    logic [5:0] cc; // cal counter
    logic [4:0] ov; // outvalue counter

    // a and w loading counter : one 16-row tile each, per (mt,nt,kt)
    logic [`ROW_IDX_W:0] a_cnt;
    logic [4:0] w_cnt;

    // descriptor-held fusion mode
    logic fuse_in_q, fuse_out_q;

    // load bookkeeping. Activation is now streamed one k-tile at a time for every
    // (mt,nt,kt) instead of caching all of A[16 x K] : that is what decouples UB
    // depth from K and lets K_MAX be 4096 with only two 16-deep macros.
    wire load_a = !fuse_in_q; // a fused job reads its activation straight from OBUF
    wire a_done = !load_a || (a_cnt == `ARRAY_S); // activation tile loaded (or not needed)
    wire w_done = (w_cnt == `ARRAY_S); // weight load complete (16 beats)
    wire a_beat = (state == LOADING) && a_valid && load_a && (a_cnt != `ARRAY_S); // accepted A beat
    wire w_beat = (state == LOADING) && w_valid && (w_cnt != `ARRAY_S); // accepted W beat

    // psum de-skew shift chain : {en,row} shifted one column per cycle (45-deg de-skew)
    logic en_chain [0:`ARRAY_S-1];
    logic [`ROW_IDX_W-1:0] row_chain [0:`ARRAY_S-1];
    wire res_valid = (state == CAL) && (cc >= RESULT_LAT) && (cc < RESULT_LAT + ACT_N); // column-0 result window
    wire [`ROW_IDX_W-1:0] res_row = cc - RESULT_LAT; // output row column-0 is emitting

    // WMEM write strobe : delayed 1 cycle to align with WPU's registered encode output
    logic wmem_wr_q;

    // next-state (comb)
    always_comb begin
        case (state)
            // 'IDLE' -> 'LOADING' when cmd_valid = 1
            IDLE: nx_state = (cmd_valid)? LOADING : IDLE;

            // 'LOADING' (Load one w and a tile into memory) -> 'PRELOAD_W' when w and a done
            LOADING : nx_state = (a_done && w_done)? PRELOAD_W : LOADING;

            // 'PRELOAD_W' (Preload weight into rsa and csa) -> 'CAL' when finish preloading
            PRELOAD_W : nx_state = (pw == `ARRAY_S)? CAL : PRELOAD_W;

            // 'CAL' (Matrix Multiplication) -> 'LOADING' when 1 tile finished
            // 'CAL' (Matrix Multiplication) -> 'OUT' when all k tile finished
            CAL: begin
                if(cc == CAL_TOTAL-1) nx_state = (kt != k_tiles-1)? LOADING : OUT;
                else nx_state = CAL;
            end

            // 'OUT' -> 'LOADING' when matrix Multiplication do not finish
            // 'OUT' -> 'DONE'
            OUT: begin
                if (r_ready && ov == `ARRAY_S-1) nx_state = (nt != n_tiles-1 || mt != m_tiles-1)? LOADING : DONE;
                else nx_state = OUT;
            end

            DONE : nx_state = IDLE;
            default : nx_state = IDLE;
        endcase
    end

    // state register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else state <= nx_state;
    end

    // psum de-skew chain : lane 0 = column-0 result, lane j = lane j-1 delayed 1 cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int j = 0; j < `ARRAY_S; j++) begin
                en_chain[j] <= 1'b0;
                row_chain[j] <= 0;
            end
        end
        else begin
            en_chain[0] <= res_valid;
            row_chain[0] <= res_row;
            for(int j = 1; j < `ARRAY_S; j++) begin
                en_chain[j] <= en_chain[j-1];
                row_chain[j] <= row_chain[j-1];
            end
        end
    end

    // counters (descriptor latch / load beats / phase / tile index)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_tiles <= 0;
            k_tiles <= 0;
            n_tiles <= 0;
            quant_mult_q <= '0;
            quant_shift_q <= '0;
            fuse_in_q <= 1'b0;
            fuse_out_q <= 1'b0;
            mt <= 0;
            nt <= 0;
            kt <= 0;
            a_cnt <= 0;
            w_cnt <= 0;
            pw <= 0;
            cc <= 0;
            ov <= 0;
            wmem_wr_q <= 1'b0;
        end
        else begin
            // sample the matrix configuration when cmd_valid and reset the counter
            if (state == IDLE && cmd_valid) begin
                m_tiles <= dim_m >> $clog2(`ARRAY_S);
                k_tiles <= dim_k >> $clog2(`ARRAY_S);
                n_tiles <= dim_n >> $clog2(`ARRAY_S);
                quant_mult_q <= quant_mult;
                quant_shift_q <= quant_shift;
                fuse_in_q <= fuse_in;
                fuse_out_q <= fuse_out;
                mt <= 0;
                nt <= 0;
                kt <= 0;
            end

            // load beat counters for weight and activation (one tile each)
            if (state == LOADING) begin
                if (a_beat) a_cnt <= a_cnt + 1;
                if (w_beat) w_cnt <= w_cnt + 1;
            end
            else begin
                a_cnt <= 0;
                w_cnt <= 0;
            end

            // WMEM write strobe : 1-cycle delay to match WPU's registered encode latency
            wmem_wr_q <= w_beat;

            // phase counters
            case (state)
                PRELOAD_W : pw <= (pw == `ARRAY_S) ? 5'd0 : pw + 5'd1;
                CAL : cc <= (cc == CAL_TOTAL-1) ? 6'd0 : cc + 6'd1;
                OUT : if (r_ready) ov <= (ov == `ARRAY_S-1) ? 5'd0 : ov + 5'd1;
                default : begin
                    pw <= 0;
                    cc <= 0;
                end
            endcase

            // tile-index updates on the block-advancing transitions
            if (state == CAL && cc == CAL_TOTAL-1 && kt != k_tiles-1) kt <= kt + 1;
            if (state == OUT && r_ready && ov == `ARRAY_S-1) begin
                if (nt != n_tiles-1) begin
                    nt <= nt + 1;
                    kt <= 0; // next output column block
                end
                else if (mt != m_tiles-1) begin
                    mt <= mt + 1;
                    nt <= 0;
                    kt <= 0; // next A row block
                end
            end
        end
    end

    // output control (comb). Defaults first to avoid latches.
    always_comb begin
        // status / external handshake
        cmd_ready = (state == IDLE);
        busy = (state != IDLE);
        done = (state == DONE);

        a_ready = (state == LOADING) && load_a && (a_cnt != `ARRAY_S);
        w_ready = (state == LOADING) && (w_cnt != `ARRAY_S);
        r_valid = (state == OUT);

        // weight write path (WPU) / WMEM
        wpu_wmem_w = w_beat; // encode only on an accepted weight beat
        if (state == LOADING) wmem_addr = `WMEM_ADDR_W'(w_cnt); // zero-extend 5-bit w_cnt (avoid OOB bit)
        else if (state == PRELOAD_W && pw <= `ARRAY_S-1) wmem_addr = (`ARRAY_S-1) - pw; // read rows 15..0 (pw=0..15)
        else wmem_addr = '0;

        // WMEM active on a (delayed) weight write, or a preload read
        wmem_en_c = wmem_wr_q || (state == PRELOAD_W && pw <= `ARRAY_S-1);
        wmem_en_w = wmem_wr_q; // write strobe delayed to WPU output ; 0 during preload (read)

        // UB : activation streaming window. The tile loaded in LOADING(kt) is the
        // tile read back in CAL(kt), so write and read select the same half ; the
        // kt[0] parity keeps the two halves alive for a future load/compute overlap.
        // A fused job never fills the window, so keep its read port idle too --
        // otherwise the UB macros toggle every CAL cycle for data nobody selects.
        ub_addr_w = a_beat ? {kt[0], a_cnt[`ROW_IDX_W-1:0]} : '0;
        ub_en_w = a_beat;
        ub_addr_r = (!fuse_in_q && state == CAL && cc < ACT_N) ? {kt[0], cc[`ROW_IDX_W-1:0]} : '0;
        ub_en_r = !fuse_in_q && (state == CAL) && (cc < ACT_N);

        // OBUF : layer fusion.
        //  write -> result row m of block (mt,nt) at [nt*16 + m]  (needs M<=16, N<=OBUF_D)
        //  read  -> next layer's activation row (kt,m) at [kt*16 + m] (needs K<=OBUF_D)
        // The range guards stop an over-sized job from wrapping and corrupting the
        // buffer ; PATTERN's SVA flags the illegal descriptor.
        obuf_addr_w = {nt[OBUF_TSEL_W-1:0], ov[`ROW_IDX_W-1:0]};
        obuf_en_w = fuse_out_q && (state == OUT) && r_ready && (nt < `OBUF_TILES);
        obuf_addr_r = (state == CAL && cc < ACT_N) ? {kt[OBUF_TSEL_W-1:0], cc[`ROW_IDX_W-1:0]} : '0;
        obuf_en_r = fuse_in_q && (state == CAL) && (cc < ACT_N) && (kt < `OBUF_TILES);
        act_sel = fuse_in_q;

        // arrays / skew
        // weight_valid aligned to the arriving (1-cycle-late) WMEM data : pw=1..16
        weight_valid = (state == PRELOAD_W) && (pw != 0);
        // UB / OBUF are synchronous-read memories: cc=0 launches the first
        // read, and the corresponding activation arrives one cycle later.
        // Do not inject the invalid first Q value into the RSA skew pipeline;
        // keep advancing for the remaining CAL cycles so the array still drains.
        skew_en = (state == CAL) && (cc >= 1);
        // clear compensation state at the start of each tile's weight load (before cout_valid pulses)
        comp_clr = (state == LOADING) && (w_cnt == 0);
        csa_cw_valid = (state == PRELOAD_W) && (pw != 0);
        // activation reaches CSA 1 cycle late (UB read latency) : valid cc=1..ACT_N -> rows 0..15
        csa_act_valid = (state == CAL) && (cc >= 1) && (cc <= ACT_N);

        // accumulator (ACC)
        acc_clr = (state == PRELOAD_W) && (kt == 0); // clear block before first kt

        // psum accumulate : de-skew chain drives per-column enable + row
        for(int j = 0; j < `ARRAY_S; j++) begin
            acc_wr_en[j] = en_chain[j];
            acc_wr_row[`ROW_IDX_W*j +: `ROW_IDX_W] = row_chain[j];
        end

        // compensation : fires every k-tile's CAL (ACC gates per-column by cout_valid ;
        // each column's single non-MSR-4 sits in exactly one k-tile, compensated there once)
        cacc_en = (state == CAL) && (cc >= CACC_LAT) && (cc < CACC_LAT + ACT_N);
        cacc_wr_row = cc - CACC_LAT; // output row m being compensated

        // read-out : sweep rows during OUT
        acc_rd_en = (state == OUT);
        acc_rd_row = ov;
    end

endmodule
