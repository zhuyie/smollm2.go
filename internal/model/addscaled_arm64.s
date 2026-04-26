//go:build arm64

#include "textflag.h"

// func addScaledF32ARM64(dst, src []float32, scale float32)
TEXT ·addScaledF32ARM64(SB), NOSPLIT, $0-52
	MOVD	dst_base+0(FP), R0
	MOVD	dst_len+8(FP), R2
	MOVD	src_base+24(FP), R1
	FMOVS	scale+48(FP), F0

	VDUP	V0.S[0], V0.S4

	LSR	$3, R2, R3
	CBZ	R3, addscaled_rem4

addscaled_loop8:
	MOVD	R0, R4
	VLD1.P	16(R1), [V1.S4]
	VLD1.P	16(R1), [V2.S4]
	VLD1.P	16(R0), [V3.S4]
	VLD1	(R0), [V4.S4]
	WORD	$0x4E21CC03 // FMLA V3.4S, V0.4S, V1.4S
	WORD	$0x4E22CC04 // FMLA V4.4S, V0.4S, V2.4S
	VST1.P	[V3.S4], 16(R4)
	VST1	[V4.S4], (R4)
	ADD	$16, R0
	SUB	$1, R3
	CBNZ	R3, addscaled_loop8

addscaled_rem4:
	AND	$7, R2, R3
	LSR	$2, R3, R3
	CBZ	R3, addscaled_done

	VLD1.P	16(R1), [V1.S4]
	VLD1	(R0), [V3.S4]
	WORD	$0x4E21CC03 // FMLA V3.4S, V0.4S, V1.4S
	VST1.P	[V3.S4], 16(R0)

addscaled_done:
	RET
