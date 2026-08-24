#======================================================
#         Design Flow MakeFile (Author: Marco)
#======================================================
# Run simulation from src/01_RTL.
# Run synthesis from src/02_SYN.
#======================================================
.PHONY: gen_suite check_replay prepare_vectors rtl_regress model_check model_compile model_emit model_replay rtl_mnist resnet_export resnet_compile resnet_emit resnet_replay resnet_evaluate rtl_resnet_block vcs_rtl vcs_rtl_dump vcs_gate syn clean

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
MODEL ?= compiler/artifacts/mnist/mnist_mlp.json
COMPILED_MODEL ?= compiler/build/compiled_model.json
MODEL_BATCH ?= 32
RESNET_MODEL ?= compiler/artifacts/resnet18_layer2_0/mnist_resnet18_layer2_0.json
RESNET_COMPILED ?= compiler/build/resnet18_layer2_0/compiled_model.json
RESNET_BATCH ?= 1
RESNET_EXPORT_ARGS ?=

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
	cd ../00_TESTBED && python3 compiler/check_compiler.py

model_compile:
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py compile ${MODEL} --outdir compiler/build

model_emit: model_compile
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py emit-test ${COMPILED_MODEL} --inputs ${MODEL} --batch-size ${MODEL_BATCH}

model_replay: model_emit
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py replay pattern

rtl_mnist: model_replay
	${VCS_RTL}

resnet_export:
	cd ../00_TESTBED && python3 compiler/models/mnist_resnet18.py ${RESNET_EXPORT_ARGS}

resnet_compile:
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py compile ${RESNET_MODEL} --outdir compiler/build/resnet18_layer2_0

resnet_emit: resnet_compile
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py emit-test ${RESNET_COMPILED} --inputs ${RESNET_MODEL} --batch-size ${RESNET_BATCH}

resnet_replay: resnet_emit
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py replay pattern

resnet_evaluate:
	cd ../00_TESTBED && python3 compiler/tpu_compiler.py evaluate ${RESNET_MODEL}

rtl_resnet_block: resnet_replay
	${VCS_RTL}

vcs_rtl_dump:
	${VCS_RTL_DUMP}

vcs_gate:
	${VCS_GATE}

syn:
	${DC_shell}

clean:
	rm -rf simv* csrc *.daidir ucli.key vc_hdrs.h novas* verdiLog *.fsdb *.vpd
