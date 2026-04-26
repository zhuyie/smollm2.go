//go:build arm64

#include "textflag.h"

// func addScaledF32ARM64(dst, src []float32, scale float32)
TEXT ·addScaledF32ARM64(SB), NOSPLIT, $0-52
	MOVD	dst_base+0(FP), R0
	MOVD	dst_len+8(FP), R2
	MOVD	src_len+32(FP), R3
	CMP	R3, R2
	CSEL	LT, R2, R3, R2 // R2 = min(len(dst), len(src))
	MOVD	src_base+24(FP), R1
	FMOVS	scale+48(FP), F0

	VDUP	V0.S[0], V0.S4

	LSR	$2, R2, R3
	CBZ	R3, addscaled_done

addscaled_loop4:
	VLD1.P	16(R1), [V1.S4]
	VLD1	(R0), [V2.S4]
	WORD	$0x4E21CC02 // FMLA V2.4S, V0.4S, V1.4S
	VST1.P	[V2.S4], 16(R0)
	SUB	$1, R3
	CBNZ	R3, addscaled_loop4

addscaled_done:
	RET
