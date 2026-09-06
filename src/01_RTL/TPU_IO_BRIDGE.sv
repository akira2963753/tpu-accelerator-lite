/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TPU_IO_BRIDGE.sv
* Project:      tpu-accelerator-lite
* Module:       TPU_IO_BRIDGE
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/
`include "define.vh"

module TPU_IO_BRIDGE (
    input logic clk,
    input logic rst_n,

    input logic [`HOST_DATA_W-1:0] host_data_i,
    input logic [`HOST_TYPE_W-1:0] host_type,
    input logic host_valid,
    output logic host_ready,

    output logic [`HOST_DATA_W-1:0] host_data_o,
    output logic host_out_valid,
    input logic host_out_ready,

    output logic core_cmd_valid,
    input logic core_cmd_ready,
    output logic [`CMD_DESC_W-1:0] core_cmd_desc,

    output logic core_w_valid,
    input logic core_w_ready,
    output logic [`W_RAW_BW-1:0] core_w_data,

    output logic core_a_valid,
    input logic core_a_ready,
    output logic [`A_BW-1:0] core_a_data,

    input logic core_r_valid,
    output logic core_r_ready,
    input logic [`R_BW-1:0] core_r_data,

    input logic core_busy,
    input logic core_done,
    output logic host_busy,
    output logic host_done
);

    localparam int INPUT_BEAT_W = $clog2(`HOST_CMD_BEATS);
    localparam int OUTPUT_BEAT_W = $clog2(`HOST_RESULT_BEATS);

    logic [`CMD_DESC_W-1:0] input_buffer;
    logic [INPUT_BEAT_W-1:0] input_beat;
    logic [`HOST_TYPE_W-1:0] active_type, pending_type;
    logic pending_valid;
    logic valid_host_type;
    logic last_input_beat;
    logic pending_ready;
    logic input_accept;

    logic [`R_BW-1:0] output_buffer;
    logic [OUTPUT_BEAT_W-1:0] output_beat;
    logic output_valid_q;
    logic host_done_q;

    //=============================================================
    //                    Host Input Interface
    //=============================================================
    always_comb begin : HOST_INPUT_CONTROL
        valid_host_type = 1'b1;
        last_input_beat = 1'b0;
        case(host_type)
            `HOST_TYPE_CMD: last_input_beat = (input_beat == `HOST_CMD_BEATS-1);
            `HOST_TYPE_WEIGHT: last_input_beat = (input_beat == `HOST_WEIGHT_BEATS-1);
            `HOST_TYPE_ACTIVATION: last_input_beat = (input_beat == `HOST_ACTIVATION_BEATS-1);
            default: valid_host_type = 1'b0;
        endcase
    end

    always_comb begin : PENDING_CONTROL
        case(pending_type)
            `HOST_TYPE_CMD: pending_ready = core_cmd_ready;
            `HOST_TYPE_WEIGHT: pending_ready = core_w_ready;
            `HOST_TYPE_ACTIVATION: pending_ready = core_a_ready;
            default: pending_ready = 1'b0;
        endcase
    end

    assign host_ready = !pending_valid && valid_host_type &&
                        ((input_beat == 0) || (host_type == active_type));
    assign input_accept = host_valid && host_ready;

    assign core_cmd_valid = pending_valid && (pending_type == `HOST_TYPE_CMD);
    assign core_cmd_desc = input_buffer;
    assign core_w_valid = pending_valid && (pending_type == `HOST_TYPE_WEIGHT);
    assign core_w_data = input_buffer[`W_RAW_BW-1:0];
    assign core_a_valid = pending_valid && (pending_type == `HOST_TYPE_ACTIVATION);
    assign core_a_data = input_buffer[`A_BW-1:0];

    always_ff @(posedge clk or negedge rst_n) begin : HOST_INPUT_BUFFER
        if(!rst_n) begin
            input_buffer <= '0;
            input_beat <= '0;
            active_type <= `HOST_TYPE_CMD;
            pending_type <= `HOST_TYPE_CMD;
            pending_valid <= 1'b0;
        end
        else begin
            if(pending_valid && pending_ready) pending_valid <= 1'b0;

            if(input_accept) begin
                input_buffer[input_beat*`HOST_DATA_W +: `HOST_DATA_W] <= host_data_i;
                if(input_beat == 0) active_type <= host_type;

                if(last_input_beat) begin
                    input_beat <= '0;
                    pending_type <= host_type;
                    pending_valid <= 1'b1;
                end
                else input_beat <= input_beat + 1'b1;
            end
        end
    end

    //=============================================================
    //                   Host Output Interface
    //=============================================================
    assign core_r_ready = !output_valid_q;
    assign host_out_valid = output_valid_q;
    assign host_data_o = output_buffer[output_beat*`HOST_DATA_W +: `HOST_DATA_W];

    always_ff @(posedge clk or negedge rst_n) begin : HOST_OUTPUT_BUFFER
        if(!rst_n) begin
            output_buffer <= '0;
            output_beat <= '0;
            output_valid_q <= 1'b0;
        end
        else begin
            if(core_r_valid && core_r_ready) begin
                output_buffer <= core_r_data;
                output_beat <= '0;
                output_valid_q <= 1'b1;
            end
            else if(host_out_valid && host_out_ready) begin
                if(output_beat == `HOST_RESULT_BEATS-1) begin
                    output_beat <= '0;
                    output_valid_q <= 1'b0;
                end
                else output_beat <= output_beat + 1'b1;
            end
        end
    end

    //=============================================================
    //                       Host Status
    //=============================================================
    assign host_busy = core_busy;
    assign host_done = host_done_q;

    always_ff @(posedge clk or negedge rst_n) begin : HOST_DONE_STATUS
        if(!rst_n) host_done_q <= 1'b0;
        else if(input_accept && last_input_beat && (host_type == `HOST_TYPE_CMD)) host_done_q <= 1'b0;
        else if(core_done) host_done_q <= 1'b1;
    end

endmodule
