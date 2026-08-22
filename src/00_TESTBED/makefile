#======================================================
#         Design Flow MakeFile (Author: Marco)
#======================================================
# Run simulation from src/01_RTL.
# Run synthesis from src/02_SYN.
#======================================================
.PHONY: gen_suite check_replay prepare_vectors rtl_regress model_check model_compile model_replay rtl_mnist vcs_rtl vcs_rtl_dump vcs_gate syn clean

#======================================================
#              Set your desired file names
#======================================================
top_design=TPU
source_file="file.f"
TIMESCALE="-timescale=1ns/1ps"
syn_tcl="syn16.tcl"
syn_log="syn.log"
PROFILE ?= regression
SIM_ARGS ?=
MODEL ?= model/artifacts/mnist/mnist_mlp.json
MODEL_BATCH ?= 32

#======================================================
#                          VCS
#======================================================
VCS_RTL = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+notimingchecks \
	${SIM_ARGS}

# Use DUMP only when waves are needed.
VCS_RTL_DUMP = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+define+DUMP \
	+notimingchecks

VCS_GATE = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+neg_tchk \
	+define+GATE

#======================================================
#                   Design Compiler
#======================================================
DC_shell = dc_shell -f ${syn_tcl} | tee ${syn_log}

vcs_rtl:
	${VCS_RTL}

gen_suite:
	cd ../00_TESTBED && python3 directed/gen_suite.py --profile ${PROFILE}

check_replay:
	cd ../00_TESTBED && python3 directed/check_replay.py --profile ${PROFILE}

prepare_vectors: gen_suite check_replay

rtl_regress: prepare_vectors vcs_rtl

model_check:
	cd ../00_TESTBED && python3 model/check_compiler.py

model_compile:
	cd ../00_TESTBED && python3 model/tpu_compiler.py compile ${MODEL} --batch-size ${MODEL_BATCH}

model_replay:
	cd ../00_TESTBED && python3 model/tpu_compiler.py replay pattern

rtl_mnist: model_compile model_replay vcs_rtl

vcs_rtl_dump:
	${VCS_RTL_DUMP}

vcs_gate:
	${VCS_GATE}

syn:
	${DC_shell}

clean:
	rm -rf simv* csrc *.daidir ucli.key vc_hdrs.h novas* verdiLog *.fsdb *.vpd
