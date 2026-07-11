#======================================================
#         Design Flow MakeFile (Author: Marco)         
#======================================================
.PHONY: vcs_rtl vcs_gate syn

#======================================================
#              Set your desired file names
#======================================================
top_design=Handshaker
source_file="file.f"
TIMESCALE="-timescale=1ns/1ps"
syn_tcl="syn90.tcl"
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