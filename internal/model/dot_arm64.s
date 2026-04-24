//go:build arm64

#include "textflag.h"

// func dotF32ARM64(a, b []float32) float32
// Based on the same ABI shape and reduction order used in github.com/tphakala/simd.
TEXT ·dotF32ARM64(SB), NOSPLIT, $0-52
	MOVD	a_base+0(FP), R0
	MOVD	a_len+8(FP), R2
	MOVD	b_len+32(FP), R3
	CMP	R3, R2
	CSEL	LT, R2, R3, R2 // R2 = min(len(a), len(b))
	MOVD	b_base+24(FP), R1

	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16

	LSR	$3, R2, R3
	CBZ	R3, dot_rem4

dot_loop8:
	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R0), [V3.S4]
	VLD1.P	16(R1), [V4.S4]
	VLD1.P	16(R1), [V5.S4]
	WORD	$0x4E24CC40 // FMLA V0.4S, V2.4S, V4.4S
	WORD	$0x4E25CC61 // FMLA V1.4S, V3.4S, V5.4S
	SUB	$1, R3
	CBNZ	R3, dot_loop8

	WORD	$0x4E21D400 // FADD V0.4S, V0.4S, V1.4S

dot_rem4:
	AND	$7, R2, R3
	LSR	$2, R3, R4
	CBZ	R4, dot_rem1

	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R1), [V4.S4]
	WORD	$0x4E24CC40 // FMLA V0.4S, V2.4S, V4.4S

dot_rem1:
	AND	$3, R3, R4
	CBZ	R4, dot_reduce

	WORD	$0x6E20D400 // FADDP V0.4S, V0.4S, V0.4S
	WORD	$0x7E30D800 // FADDP S0, V0.2S

dot_scalar:
	FMOVS	(R0), F2
	FMOVS	(R1), F4
	FMADDS	F4, F0, F2, F0 // F0 = F2*F4 + F0
	ADD	$4, R0
	ADD	$4, R1
	SUB	$1, R4
	CBNZ	R4, dot_scalar

	FMOVS	F0, ret+48(FP)
	RET

dot_reduce:
	WORD	$0x6E20D400 // FADDP V0.4S, V0.4S, V0.4S
	WORD	$0x7E30D800 // FADDP S0, V0.2S
	FMOVS	F0, ret+48(FP)
	RET
