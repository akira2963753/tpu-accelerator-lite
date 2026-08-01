#======================================================
#         Design Flow MakeFile (Author: Marco)         
#======================================================
# Shared by every per-block env under 10_VERI/<BLOCK>/.
# Run from the block's own 00_TESTBED / 02_SYN / 03_GATE :
#     make -f ../../makefile vcs_rtl
# syn_tcl is overridable per block :  make -f ../../makefile syn syn_tcl=syn90.tcl
#======================================================
.PHONY: vcs_rtl vcs_gate syn clean

#======================================================
#              Set your desired file names
#======================================================
top_design=TPU
source_file="file.f"
TIMESCALE="-timescale=1ns/1ps"
syn_tcl="syn16.tcl"
syn_log="syn.log"

#======================================================
#                          VCS
#======================================================
VCS_RTL = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+define+SVA \
	+notimingchecks

VCS_GATE = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+neg_tchk \
	+define+SVA \
	+define+GATE

#======================================================
#                   Design Compiler
#======================================================
DC_shell = dc_shell -f ${syn_tcl} | tee ${syn_log}

vcs_rtl:
	${VCS_RTL}

vcs_gate:
	${VCS_GATE}

syn:
	${DC_shell}

clean:
	rm -rf simv* csrc *.daidir ucli.key vc_hdrs.h novas* verdiLog *.fsdb *.vpd
