#======================================================
#         Design Flow MakeFile (Author: Marco)
#======================================================
# Simulation runs from src/01_RTL :
#     make -f ../00_TESTBED/makefile vcs_rtl
# file.f uses ./*.sv and PATTERN reads ../00_TESTBED/pattern/tNN_*.dat, so that
# cwd is required. Synthesis runs from src/02_SYN (its own file.f + Netlist).
#======================================================
.PHONY: vcs_rtl vcs_rtl_dump vcs_gate syn clean

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

# waves : the suite is ~3.2M cycles (26M with --big), so FSDB dumping is opt-in
VCS_RTL_DUMP = vcs ${TIMESCALE} \
	-full64 \
	-debug_access+all \
	-R \
	+v2k \
	-f ${source_file} \
	+define+SVA \
	+define+DUMP \
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

vcs_rtl_dump:
	${VCS_RTL_DUMP}

vcs_gate:
	${VCS_GATE}

syn:
	${DC_shell}

clean:
	rm -rf simv* csrc *.daidir ucli.key vc_hdrs.h novas* verdiLog *.fsdb *.vpd
