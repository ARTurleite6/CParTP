	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 15, 0
	.globl	__Z10add_sourceiiiPfS_f         ; -- Begin function _Z10add_sourceiiiPfS_f
	.p2align	2
__Z10add_sourceiiiPfS_f:                ; @_Z10add_sourceiiiPfS_f
	.cfi_startproc
; %bb.0:
                                        ; kill: def $s0 killed $s0 def $q0
	add	w8, w0, #2
	add	w9, w1, #2
	mul	w8, w9, w8
	add	w9, w2, #2
	mul	w8, w8, w9
	cmp	w8, #1
	b.lt	LBB0_9
; %bb.1:
	cmp	w8, #16
	b.lo	LBB0_6
; %bb.2:
	lsl	x9, x8, #2
	add	x10, x3, x9
	add	x9, x4, x9
	cmp	x9, x3
	ccmp	x10, x4, #0, hi
	b.hi	LBB0_6
; %bb.3:
	and	x9, x8, #0xfffffff0
	dup.4s	v1, v0[0]
	add	x10, x3, #32
	add	x11, x4, #32
	mov	x12, x9
LBB0_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x11, #-32]
	ldp	q4, q5, [x11], #64
	ldp	q6, q7, [x10, #-32]
	ldp	q16, q17, [x10]
	fmla.4s	v6, v2, v1
	fmla.4s	v7, v3, v1
	fmla.4s	v16, v4, v1
	fmla.4s	v17, v5, v1
	stp	q6, q7, [x10, #-32]
	stp	q16, q17, [x10], #64
	subs	x12, x12, #16
	b.ne	LBB0_4
; %bb.5:
	cmp	x9, x8
	b.ne	LBB0_7
	b	LBB0_9
LBB0_6:
	mov	x9, #0
LBB0_7:
	lsl	x11, x9, #2
	add	x10, x3, x11
	add	x11, x4, x11
	sub	x8, x8, x9
LBB0_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x11], #4
	ldr	s2, [x10]
	fmadd	s1, s0, s1, s2
	str	s1, [x10], #4
	subs	x8, x8, #1
	b.ne	LBB0_8
LBB0_9:
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__Z7set_bndiiiiPf               ; -- Begin function _Z7set_bndiiiiPf
	.p2align	2
__Z7set_bndiiiiPf:                      ; @_Z7set_bndiiiiPf
	.cfi_startproc
; %bb.0:
	stp	x20, x19, [sp, #-16]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset w19, -8
	.cfi_offset w20, -16
                                        ; kill: def $w0 killed $w0 def $x0
	cmp	w0, #1
	b.lt	LBB1_9
; %bb.1:
	cmp	w1, #1
	b.lt	LBB1_21
; %bb.2:
	add	w8, w0, #2
	add	w9, w1, #2
	mul	w9, w9, w8
	mul	w10, w9, w2
	add	w9, w9, w10
	sxtw	x12, w10
	sxtw	x10, w9
	add	w15, w1, #1
	add	w9, w0, #1
	add	x10, x10, x8
	lsl	x10, x10, #2
	add	x10, x10, #4
	lsl	x11, x8, #2
	add	x12, x12, x8
	lsl	x12, x12, #2
	add	x12, x12, #4
	add	x13, x11, #4
	add	w14, w1, #3
	mov	w16, #1
	madd	w14, w14, w8, w16
	sub	x15, x15, #1
	mov	w16, #1
	cmp	w3, #3
	b.ne	LBB1_6
LBB1_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_4 Depth 2
	mov	x17, x15
	mov	x5, x14
	mov	x6, x4
LBB1_4:                                 ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	s0, [x4, w5, sxtw #2]
	fneg	s0, s0
	str	s0, [x6, x13]
	ldr	s0, [x6, x12]
	fneg	s0, s0
	str	s0, [x6, x10]
	add	x6, x6, x11
	add	w5, w5, w8
	subs	x17, x17, #1
	b.ne	LBB1_4
; %bb.5:                                ;   in Loop: Header=BB1_3 Depth=1
	add	x16, x16, #1
	add	x10, x10, #4
	add	x12, x12, #4
	add	x13, x13, #4
	add	w14, w14, #1
	cmp	x16, x9
	b.ne	LBB1_3
	b	LBB1_9
LBB1_6:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_7 Depth 2
	mov	x17, x15
	mov	x5, x14
	mov	x6, x4
LBB1_7:                                 ;   Parent Loop BB1_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	s0, [x4, w5, sxtw #2]
	str	s0, [x6, x13]
	ldr	s0, [x6, x12]
	str	s0, [x6, x10]
	add	x6, x6, x11
	add	w5, w5, w8
	subs	x17, x17, #1
	b.ne	LBB1_7
; %bb.8:                                ;   in Loop: Header=BB1_6 Depth=1
	add	x16, x16, #1
	add	x10, x10, #4
	add	x12, x12, #4
	add	x13, x13, #4
	add	w14, w14, #1
	cmp	x16, x9
	b.ne	LBB1_6
LBB1_9:
	cmp	w1, #1
	b.lt	LBB1_20
; %bb.10:
	cmp	w2, #1
	b.lt	LBB1_20
; %bb.11:
	add	w8, w0, #2
	add	w9, w1, #2
	mul	w9, w9, w8
	sxtw	x9, w9
	add	w14, w2, #1
	sxtw	x11, w8
	add	w10, w1, #1
	add	x17, x9, x11
	lsl	x11, x11, #2
	lsl	x12, x9, #2
	add	w13, w1, #3
	cmp	w3, #1
	b.ne	LBB1_16
; %bb.12:
	mul	w13, w13, w8
	sub	x14, x14, #1
	mov	w15, #1
	add	w16, w9, w0, lsl #1
	add	x17, x4, x17, lsl #2
LBB1_13:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_14 Depth 2
	mov	w5, #0
	mov	x6, x14
	mov	x7, x17
LBB1_14:                                ;   Parent Loop BB1_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	w19, w13, w5
	add	x19, x4, w19, sxtw #2
	ldr	s0, [x19, #4]
	fneg	s0, s0
	str	s0, [x7]
	add	w19, w16, w5
	add	w20, w19, #2
	ldr	s0, [x4, w20, sxtw #2]
	add	w19, w19, #3
	fneg	s0, s0
	str	s0, [x4, w19, sxtw #2]
	add	w5, w5, w9
	add	x7, x7, x12
	subs	x6, x6, #1
	b.ne	LBB1_14
; %bb.15:                               ;   in Loop: Header=BB1_13 Depth=1
	add	x15, x15, #1
	add	w16, w16, w8
	add	x17, x17, x11
	add	w13, w13, w8
	cmp	x15, x10
	b.ne	LBB1_13
	b	LBB1_20
LBB1_16:
	mul	w13, w13, w8
	sub	x14, x14, #1
	mov	w15, #1
	add	w16, w9, w0, lsl #1
	add	x17, x4, x17, lsl #2
LBB1_17:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_18 Depth 2
	mov	w5, #0
	mov	x6, x14
	mov	x7, x17
LBB1_18:                                ;   Parent Loop BB1_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	w19, w13, w5
	add	x19, x4, w19, sxtw #2
	ldr	s0, [x19, #4]
	str	s0, [x7]
	add	w19, w16, w5
	add	w20, w19, #2
	ldr	s0, [x4, w20, sxtw #2]
	add	w19, w19, #3
	str	s0, [x4, w19, sxtw #2]
	add	w5, w5, w9
	add	x7, x7, x12
	subs	x6, x6, #1
	b.ne	LBB1_18
; %bb.19:                               ;   in Loop: Header=BB1_17 Depth=1
	add	x15, x15, #1
	add	w16, w16, w8
	add	x17, x17, x11
	add	w13, w13, w8
	cmp	x15, x10
	b.ne	LBB1_17
LBB1_20:
	cmp	w0, #0
	b.le	LBB1_27
LBB1_21:
	add	w11, w0, #2
	add	w8, w1, #2
	mul	w9, w8, w11
	mul	w8, w11, w1
	add	w10, w11, w8
	cmp	w2, #1
	b.lt	LBB1_32
; %bb.22:
	sxtw	x14, w9
	add	w2, w2, #1
	add	w12, w0, #1
	add	x13, x14, w0, uxtw
	add	x13, x4, x13, lsl #2
	add	x13, x13, #12
	lsl	x14, x14, #2
	add	x15, x14, x4
	add	x15, x15, #4
	lsl	w17, w1, #1
	add	w16, w17, #3
	mov	w1, #1
	madd	w16, w16, w11, w1
	add	w17, w17, #2
	cmp	w3, #2
	b.ne	LBB1_28
; %bb.23:
	mul	w17, w11, w17
	orr	w17, w17, #0x1
	sub	x1, x2, #1
	mov	w2, #1
LBB1_24:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_25 Depth 2
	mov	w3, #0
	mov	x5, #0
	mov	x6, x1
LBB1_25:                                ;   Parent Loop BB1_24 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	s0, [x13, x5]
	fneg	s0, s0
	str	s0, [x15, x5]
	add	w7, w17, w3
	ldr	s0, [x4, w7, sxtw #2]
	add	w7, w16, w3
	fneg	s0, s0
	str	s0, [x4, w7, sxtw #2]
	add	x5, x5, x14
	add	w3, w3, w9
	subs	x6, x6, #1
	b.ne	LBB1_25
; %bb.26:                               ;   in Loop: Header=BB1_24 Depth=1
	add	x2, x2, #1
	add	x13, x13, #4
	add	x15, x15, #4
	add	w16, w16, #1
	add	w17, w17, #1
	cmp	x2, x12
	b.ne	LBB1_24
	b	LBB1_32
LBB1_27:
	add	w11, w0, #2
	add	w8, w1, #2
	mul	w9, w8, w11
	mul	w8, w11, w1
	add	w10, w11, w8
	b	LBB1_32
LBB1_28:
	mul	w17, w11, w17
	orr	w17, w17, #0x1
	sub	x1, x2, #1
	mov	w2, #1
LBB1_29:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_30 Depth 2
	mov	w3, #0
	mov	x5, #0
	mov	x6, x1
LBB1_30:                                ;   Parent Loop BB1_29 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	s0, [x13, x5]
	str	s0, [x15, x5]
	add	w7, w17, w3
	ldr	s0, [x4, w7, sxtw #2]
	add	w7, w16, w3
	str	s0, [x4, w7, sxtw #2]
	add	x5, x5, x14
	add	w3, w3, w9
	subs	x6, x6, #1
	b.ne	LBB1_30
; %bb.31:                               ;   in Loop: Header=BB1_29 Depth=1
	add	x2, x2, #1
	add	x13, x13, #4
	add	x15, x15, #4
	add	w16, w16, #1
	add	w17, w17, #1
	cmp	x2, x12
	b.ne	LBB1_29
LBB1_32:
	ldr	s0, [x4, w11, sxtw #2]
	ldr	s1, [x4, #4]
	fadd	s0, s1, s0
	sxtw	x12, w9
	ldr	s1, [x4, w9, sxtw #2]
	fadd	s0, s0, s1
	mov	w13, #62915
	movk	w13, #16040, lsl #16
	fmov	s1, w13
	fmul	s0, s0, s1
	str	s0, [x4]
	sxtw	x13, w0
	add	x14, x4, w0, sxtw #2
	ldr	s0, [x14]
	add	x11, x13, w11, sxtw
	add	x11, x4, x11, lsl #2
	ldr	s2, [x11, #4]
	fadd	s0, s0, s2
	add	x11, x12, x13
	add	x11, x4, x11, lsl #2
	ldr	s2, [x11, #4]
	fadd	s0, s0, s2
	fmul	s0, s0, s1
	str	s0, [x14, #4]
	add	x11, x4, w10, sxtw #2
	ldr	s0, [x11, #4]
	ldr	s2, [x4, w8, sxtw #2]
	fadd	s0, s0, s2
	add	w9, w10, w9
	ldr	s2, [x4, w9, sxtw #2]
	fadd	s0, s0, s2
	fmul	s0, s0, s1
	str	s0, [x11]
	add	x9, x13, w10, sxtw
	add	x10, x4, x9, lsl #2
	ldr	s0, [x10]
	add	x8, x13, w8, sxtw
	add	x8, x4, x8, lsl #2
	ldr	s2, [x8, #4]
	fadd	s0, s0, s2
	add	x8, x9, x12
	add	x8, x4, x8, lsl #2
	ldr	s2, [x8, #4]
	fadd	s0, s0, s2
	fmul	s0, s0, s1
	str	s0, [x10, #4]
	ldp	x20, x19, [sp], #16             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__Z9lin_solveiiiiPfS_ff         ; -- Begin function _Z9lin_solveiiiiPfS_ff
	.p2align	2
__Z9lin_solveiiiiPfS_ff:                ; @_Z9lin_solveiiiiPfS_ff
	.cfi_startproc
; %bb.0:
	stp	d9, d8, [sp, #-112]!            ; 16-byte Folded Spill
	.cfi_def_cfa_offset 112
	stp	x28, x27, [sp, #16]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	sub	sp, sp, #1152
	mov	x19, x4
	stp	w3, w0, [sp, #232]              ; 8-byte Folded Spill
	str	w2, [sp, #240]                  ; 4-byte Folded Spill
	mov	x22, x1
	cmp	w0, #1
	b.lt	LBB2_25
; %bb.1:
	cmp	w22, #1
	b.lt	LBB2_25
; %bb.2:
	ldr	w8, [sp, #240]                  ; 4-byte Folded Reload
	cmp	w8, #1
	b.lt	LBB2_25
; %bb.3:
	mov	w1, #0
	ldr	w0, [sp, #236]                  ; 4-byte Folded Reload
	add	w25, w0, #2
	add	w8, w22, #2
	mul	w8, w8, w25
	lsl	w2, w8, #3
	mov	w17, #1
	mov	w9, #1
	bfi	w9, w8, #3, #29
	str	w9, [sp, #212]                  ; 4-byte Folded Spill
	lsl	w9, w0, #3
	add	w9, w9, #16
	str	w9, [sp, #472]                  ; 4-byte Folded Spill
	lsl	w9, w0, #1
	sub	w10, w2, w8
	add	w11, w10, w9
	add	w10, w10, #1
	str	w10, [sp, #208]                 ; 4-byte Folded Spill
	lsl	w10, w8, #1
	mov	w14, #1
	add	w12, w10, w8
	add	w13, w12, w9
	bfi	w14, w12, #1, #31
	str	w14, [sp, #204]                 ; 4-byte Folded Spill
	lsl	w14, w8, #2
	add	w15, w14, w8
	add	w16, w15, w9
	add	w15, w15, #1
	str	w15, [sp, #200]                 ; 4-byte Folded Spill
	mov	w15, #1
	bfi	w15, w8, #2, #30
	str	w15, [sp, #196]                 ; 4-byte Folded Spill
	fdiv	s8, s0, s1
	bfi	w17, w8, #1, #31
	str	x17, [sp, #216]                 ; 8-byte Folded Spill
	ubfiz	x28, x25, #2, #32
	mov	w15, #36
	fmov	s0, #1.00000000
	add	x17, x28, x19
	add	x17, x17, #4
	smaddl	x15, w8, w15, x17
	stp	x15, x17, [sp, #176]            ; 16-byte Folded Spill
	sbfiz	x15, x8, #5, #32
	str	x15, [sp, #688]                 ; 8-byte Folded Spill
	add	x15, x15, x28
	add	x15, x15, #4
	add	x17, x5, x15
	add	x15, x19, x15
	stp	x15, x17, [sp, #160]            ; 16-byte Folded Spill
	mov	w15, #28
	smaddl	x15, w8, w15, x28
	add	x15, x15, #4
	add	x17, x5, x15
	add	x15, x19, x15
	stp	x15, x17, [sp, #144]            ; 16-byte Folded Spill
	mov	w15, #24
	smaddl	x15, w8, w15, x28
	add	x15, x15, #4
	add	x17, x5, x15
	str	x17, [sp, #136]                 ; 8-byte Folded Spill
	add	w17, w8, w9
	add	w17, w17, #5
	add	w10, w10, w9
	add	w10, w10, #5
	stp	w10, w17, [sp, #128]            ; 8-byte Folded Spill
	add	x10, x19, x15
	str	x10, [sp, #120]                 ; 8-byte Folded Spill
	mov	w10, #20
	smaddl	x10, w8, w10, x28
	add	x10, x10, #4
	add	w15, w13, #5
	add	w13, w14, w9
	add	w13, w13, #5
	stp	w13, w15, [sp, #112]            ; 8-byte Folded Spill
	add	x13, x5, x10
	add	x10, x19, x10
	stp	x10, x13, [sp, #96]             ; 16-byte Folded Spill
	add	x10, x28, w8, sxtw #4
	add	x10, x10, #4
	add	w14, w16, #5
	add	w13, w12, #1
	stp	w13, w14, [sp, #88]             ; 8-byte Folded Spill
	add	w12, w9, w12, lsl #1
	add	w12, w12, #5
	str	w12, [sp, #84]                  ; 4-byte Folded Spill
	add	x12, x5, x10
	add	x10, x19, x10
	stp	x10, x12, [sp, #64]             ; 16-byte Folded Spill
	fdiv	s9, s0, s1
	mov	w10, #12
	smaddl	x10, w8, w10, x28
	add	x10, x10, #4
	add	w11, w11, #5
	str	w2, [sp, #700]                  ; 4-byte Folded Spill
	add	w9, w2, w9
	add	w9, w9, #5
	stp	w9, w11, [sp, #56]              ; 8-byte Folded Spill
	add	x9, x28, w8, sxtw #3
	add	w11, w8, #1
	str	w11, [sp, #52]                  ; 4-byte Folded Spill
	add	x8, x25, w8, sxtw
	lsl	x8, x8, #2
	add	x11, x5, x10
	add	x10, x19, x10
	stp	x10, x11, [sp, #32]             ; 16-byte Folded Spill
	add	x9, x9, #4
	add	x10, x5, x9
	str	x10, [sp, #24]                  ; 8-byte Folded Spill
	add	x10, x8, x5
	add	x10, x10, #4
	add	x8, x8, x19
	add	x8, x8, #8
	stp	x8, x10, [sp, #8]               ; 16-byte Folded Spill
	add	x8, x19, x9
	str	x8, [sp]                        ; 8-byte Folded Spill
	ldr	w8, [sp, #240]                  ; 4-byte Folded Reload
	mov	w21, w8
	ubfiz	x8, x25, #5, #32
	str	x8, [sp, #464]                  ; 8-byte Folded Spill
	mov	w23, w22
	mov	w8, w0
	str	x8, [sp, #944]                  ; 8-byte Folded Spill
	str	w22, [sp, #228]                 ; 4-byte Folded Spill
	b	LBB2_5
LBB2_4:                                 ;   in Loop: Header=BB2_5 Depth=1
	ldp	w0, w2, [sp, #236]              ; 8-byte Folded Reload
	ldp	w1, w3, [sp, #228]              ; 8-byte Folded Reload
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	ldr	w1, [sp, #244]                  ; 4-byte Folded Reload
	add	w1, w1, #1
	cmp	w1, #20
	b.eq	LBB2_26
LBB2_5:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_7 Depth 2
                                        ;       Child Loop BB2_9 Depth 3
                                        ;         Child Loop BB2_11 Depth 4
                                        ;           Child Loop BB2_12 Depth 5
                                        ;             Child Loop BB2_13 Depth 6
	str	w1, [sp, #244]                  ; 4-byte Folded Spill
	ldp	x10, x11, [sp]                  ; 16-byte Folded Reload
	ldp	x12, x13, [sp, #16]             ; 16-byte Folded Reload
	ldp	x14, x17, [sp, #32]             ; 16-byte Folded Reload
	ldp	w15, w16, [sp, #56]             ; 8-byte Folded Reload
	ldp	x0, x3, [sp, #64]               ; 16-byte Folded Reload
	ldr	w2, [sp, #92]                   ; 4-byte Folded Reload
	ldp	x4, x7, [sp, #96]               ; 16-byte Folded Reload
	ldp	w5, w6, [sp, #112]              ; 8-byte Folded Reload
	ldr	x20, [sp, #120]                 ; 8-byte Folded Reload
	ldp	w22, w24, [sp, #128]            ; 8-byte Folded Reload
	ldp	x26, x8, [sp, #136]             ; 16-byte Folded Reload
	str	x8, [sp, #448]                  ; 8-byte Folded Spill
	ldr	x8, [sp, #152]                  ; 8-byte Folded Reload
	str	x8, [sp, #440]                  ; 8-byte Folded Spill
	ldr	x8, [sp, #160]                  ; 8-byte Folded Reload
	str	x8, [sp, #432]                  ; 8-byte Folded Spill
	ldr	x8, [sp, #168]                  ; 8-byte Folded Reload
	str	x8, [sp, #424]                  ; 8-byte Folded Spill
	ldp	x8, x9, [sp, #176]              ; 16-byte Folded Reload
	str	x8, [sp, #416]                  ; 8-byte Folded Spill
	ldr	w8, [sp, #52]                   ; 4-byte Folded Reload
	str	w8, [sp, #412]                  ; 4-byte Folded Spill
	ldr	x8, [sp, #216]                  ; 8-byte Folded Reload
                                        ; kill: def $w8 killed $w8 killed $x8
	str	w8, [sp, #408]                  ; 4-byte Folded Spill
	ldp	w1, w8, [sp, #84]               ; 8-byte Folded Reload
	str	w8, [sp, #404]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #196]                  ; 4-byte Folded Reload
	str	w8, [sp, #400]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #200]                  ; 4-byte Folded Reload
	str	w8, [sp, #396]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #204]                  ; 4-byte Folded Reload
	str	w8, [sp, #392]                  ; 4-byte Folded Spill
	ldp	w8, w30, [sp, #208]             ; 8-byte Folded Reload
	str	w8, [sp, #388]                  ; 4-byte Folded Spill
	mov	w27, #1
	b	LBB2_7
LBB2_6:                                 ;   in Loop: Header=BB2_7 Depth=2
	ldr	w8, [sp, #460]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #460]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #388]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #388]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #392]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #392]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #396]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #396]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #400]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #400]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #404]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #404]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #408]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #408]                  ; 4-byte Folded Spill
	ldr	w8, [sp, #412]                  ; 4-byte Folded Reload
	add	w8, w8, #8
	str	w8, [sp, #412]                  ; 4-byte Folded Spill
	ldr	x8, [sp, #416]                  ; 8-byte Folded Reload
	add	x9, x8, #32
	ldr	x8, [sp, #424]                  ; 8-byte Folded Reload
	add	x8, x8, #32
	stp	x9, x8, [sp, #416]              ; 16-byte Folded Spill
	ldr	x8, [sp, #432]                  ; 8-byte Folded Reload
	add	x9, x8, #32
	ldr	x8, [sp, #440]                  ; 8-byte Folded Reload
	add	x8, x8, #32
	stp	x9, x8, [sp, #432]              ; 16-byte Folded Spill
	ldr	x8, [sp, #448]                  ; 8-byte Folded Reload
	add	x8, x8, #32
	str	x8, [sp, #448]                  ; 8-byte Folded Spill
	ldr	x26, [sp, #248]                 ; 8-byte Folded Reload
	add	x26, x26, #32
	ldr	w24, [sp, #256]                 ; 4-byte Folded Reload
	add	w24, w24, #8
	ldr	w22, [sp, #260]                 ; 4-byte Folded Reload
	add	w22, w22, #8
	ldp	x20, x7, [sp, #264]             ; 16-byte Folded Reload
	add	x20, x20, #32
	add	x7, x7, #32
	ldr	w6, [sp, #280]                  ; 4-byte Folded Reload
	add	w6, w6, #8
	ldr	w5, [sp, #284]                  ; 4-byte Folded Reload
	add	w5, w5, #8
	ldp	x4, x3, [sp, #288]              ; 16-byte Folded Reload
	add	x4, x4, #32
	add	x3, x3, #32
	ldr	w2, [sp, #304]                  ; 4-byte Folded Reload
	add	w2, w2, #8
	ldr	w1, [sp, #308]                  ; 4-byte Folded Reload
	add	w1, w1, #8
	ldp	x0, x17, [sp, #312]             ; 16-byte Folded Reload
	add	x0, x0, #32
	add	x17, x17, #32
	ldr	w16, [sp, #328]                 ; 4-byte Folded Reload
	add	w16, w16, #8
	ldr	w15, [sp, #332]                 ; 4-byte Folded Reload
	add	w15, w15, #8
	ldp	x14, x13, [sp, #336]            ; 16-byte Folded Reload
	add	x14, x14, #32
	add	x13, x13, #32
	ldp	x12, x11, [sp, #352]            ; 16-byte Folded Reload
	add	x12, x12, #32
	add	x11, x11, #32
	ldp	x10, x9, [sp, #368]             ; 16-byte Folded Reload
	add	x10, x10, #32
	add	x9, x9, #32
	ldr	x30, [sp, #952]                 ; 8-byte Folded Reload
	mov	x27, x30
	ldr	x8, [sp, #944]                  ; 8-byte Folded Reload
	cmp	x30, x8
	ldr	w30, [sp, #460]                 ; 4-byte Folded Reload
	b.hi	LBB2_4
LBB2_7:                                 ;   Parent Loop BB2_5 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_9 Depth 3
                                        ;         Child Loop BB2_11 Depth 4
                                        ;           Child Loop BB2_12 Depth 5
                                        ;             Child Loop BB2_13 Depth 6
	str	x27, [sp, #704]                 ; 8-byte Folded Spill
	add	x8, x27, #8
	str	x8, [sp, #952]                  ; 8-byte Folded Spill
	stp	x10, x9, [sp, #368]             ; 16-byte Folded Spill
	mov	x8, x9
	stp	x12, x11, [sp, #352]            ; 16-byte Folded Spill
	stp	x14, x13, [sp, #336]            ; 16-byte Folded Spill
	str	w15, [sp, #332]                 ; 4-byte Folded Spill
	str	w16, [sp, #328]                 ; 4-byte Folded Spill
	stp	x0, x17, [sp, #312]             ; 16-byte Folded Spill
	str	w1, [sp, #308]                  ; 4-byte Folded Spill
	str	w2, [sp, #304]                  ; 4-byte Folded Spill
	stp	x4, x3, [sp, #288]              ; 16-byte Folded Spill
	str	w5, [sp, #284]                  ; 4-byte Folded Spill
	str	w6, [sp, #280]                  ; 4-byte Folded Spill
	stp	x20, x7, [sp, #264]             ; 16-byte Folded Spill
	mov	x27, x7
	str	w22, [sp, #260]                 ; 4-byte Folded Spill
	str	w24, [sp, #256]                 ; 4-byte Folded Spill
	str	x26, [sp, #248]                 ; 8-byte Folded Spill
	ldr	x9, [sp, #448]                  ; 8-byte Folded Reload
	str	x9, [sp, #680]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #440]                  ; 8-byte Folded Reload
	str	x9, [sp, #672]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #432]                  ; 8-byte Folded Reload
	str	x9, [sp, #664]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #424]                  ; 8-byte Folded Reload
	str	x9, [sp, #656]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #416]                  ; 8-byte Folded Reload
	str	x9, [sp, #648]                  ; 8-byte Folded Spill
	ldr	w9, [sp, #412]                  ; 4-byte Folded Reload
	str	w9, [sp, #644]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #408]                  ; 4-byte Folded Reload
	str	w9, [sp, #640]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #404]                  ; 4-byte Folded Reload
	str	w9, [sp, #636]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #400]                  ; 4-byte Folded Reload
	str	w9, [sp, #632]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #396]                  ; 4-byte Folded Reload
	str	w9, [sp, #628]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #392]                  ; 4-byte Folded Reload
	str	w9, [sp, #624]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #388]                  ; 4-byte Folded Reload
	str	w9, [sp, #620]                  ; 4-byte Folded Spill
	str	w30, [sp, #460]                 ; 4-byte Folded Spill
	mov	w9, #1
	b	LBB2_9
LBB2_8:                                 ;   in Loop: Header=BB2_9 Depth=3
	ldr	w8, [sp, #472]                  ; 4-byte Folded Reload
	ldr	w30, [sp, #476]                 ; 4-byte Folded Reload
	add	w30, w30, w8
	ldr	w9, [sp, #620]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #620]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #624]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #624]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #628]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #628]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #632]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #632]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #636]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #636]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #640]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #640]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #644]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #644]                  ; 4-byte Folded Spill
	ldr	x9, [sp, #464]                  ; 8-byte Folded Reload
	ldr	x10, [sp, #648]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #648]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #656]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #656]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #664]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #664]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #672]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #672]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #680]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #680]                 ; 8-byte Folded Spill
	ldr	x26, [sp, #480]                 ; 8-byte Folded Reload
	add	x26, x26, x9
	ldr	w24, [sp, #488]                 ; 4-byte Folded Reload
	add	w24, w24, w8
	ldr	w22, [sp, #492]                 ; 4-byte Folded Reload
	add	w22, w22, w8
	ldp	x20, x27, [sp, #496]            ; 16-byte Folded Reload
	add	x20, x20, x9
	add	x27, x27, x9
	ldr	w6, [sp, #512]                  ; 4-byte Folded Reload
	add	w6, w6, w8
	ldr	w5, [sp, #516]                  ; 4-byte Folded Reload
	add	w5, w5, w8
	ldr	x4, [sp, #520]                  ; 8-byte Folded Reload
	add	x4, x4, x9
	ldr	x3, [sp, #528]                  ; 8-byte Folded Reload
	add	x3, x3, x9
	ldr	w2, [sp, #536]                  ; 4-byte Folded Reload
	add	w2, w2, w8
	ldr	w1, [sp, #540]                  ; 4-byte Folded Reload
	add	w1, w1, w8
	ldr	x0, [sp, #544]                  ; 8-byte Folded Reload
	add	x0, x0, x9
	ldr	x17, [sp, #552]                 ; 8-byte Folded Reload
	add	x17, x17, x9
	ldr	w16, [sp, #560]                 ; 4-byte Folded Reload
	add	w16, w16, w8
	ldr	w15, [sp, #564]                 ; 4-byte Folded Reload
	add	w15, w15, w8
	ldr	x14, [sp, #568]                 ; 8-byte Folded Reload
	add	x14, x14, x9
	ldr	x13, [sp, #576]                 ; 8-byte Folded Reload
	add	x13, x13, x9
	ldr	x12, [sp, #584]                 ; 8-byte Folded Reload
	add	x12, x12, x9
	ldr	x11, [sp, #592]                 ; 8-byte Folded Reload
	add	x11, x11, x9
	ldr	x10, [sp, #600]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	ldr	x8, [sp, #608]                  ; 8-byte Folded Reload
	add	x8, x8, x9
	mov	x9, x7
	cmp	x7, x23
	b.hi	LBB2_6
LBB2_9:                                 ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_7 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB2_11 Depth 4
                                        ;           Child Loop BB2_12 Depth 5
                                        ;             Child Loop BB2_13 Depth 6
	str	x9, [sp, #960]                  ; 8-byte Folded Spill
	add	x7, x9, #8
	str	x8, [sp, #608]                  ; 8-byte Folded Spill
	str	x10, [sp, #600]                 ; 8-byte Folded Spill
	str	x11, [sp, #592]                 ; 8-byte Folded Spill
	str	x12, [sp, #584]                 ; 8-byte Folded Spill
	str	x13, [sp, #576]                 ; 8-byte Folded Spill
	str	x14, [sp, #568]                 ; 8-byte Folded Spill
	str	w15, [sp, #564]                 ; 4-byte Folded Spill
	str	w16, [sp, #560]                 ; 4-byte Folded Spill
	str	x17, [sp, #552]                 ; 8-byte Folded Spill
	mov	x9, x17
	mov	x17, x13
	str	x0, [sp, #544]                  ; 8-byte Folded Spill
	mov	x13, x0
	mov	x0, x14
	str	w1, [sp, #540]                  ; 4-byte Folded Spill
	mov	x14, x1
	mov	x1, x15
	str	w2, [sp, #536]                  ; 4-byte Folded Spill
	mov	x15, x2
	mov	x2, x16
	str	x3, [sp, #528]                  ; 8-byte Folded Spill
	mov	x16, x3
	mov	x3, x9
	str	x4, [sp, #520]                  ; 8-byte Folded Spill
	mov	x9, x4
	mov	x4, x13
	str	w5, [sp, #516]                  ; 4-byte Folded Spill
	mov	x13, x5
	mov	x5, x14
	str	w6, [sp, #512]                  ; 4-byte Folded Spill
	mov	x14, x6
	stp	x20, x27, [sp, #496]            ; 16-byte Folded Spill
	mov	x6, x27
	mov	x27, x13
	mov	x13, x20
	mov	x20, x15
	str	w22, [sp, #492]                 ; 4-byte Folded Spill
	mov	x15, x22
	str	w24, [sp, #488]                 ; 4-byte Folded Spill
	mov	x22, x24
	mov	x24, x16
	str	x26, [sp, #480]                 ; 8-byte Folded Spill
	str	x26, [sp, #928]                 ; 8-byte Folded Spill
	mov	x26, x9
	mov	x16, x30
	ldr	x30, [sp, #680]                 ; 8-byte Folded Reload
	ldr	x9, [sp, #672]                  ; 8-byte Folded Reload
	str	x9, [sp, #920]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #664]                  ; 8-byte Folded Reload
	str	x9, [sp, #912]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #656]                  ; 8-byte Folded Reload
	str	x9, [sp, #904]                  ; 8-byte Folded Spill
	ldr	x9, [sp, #648]                  ; 8-byte Folded Reload
	str	x9, [sp, #896]                  ; 8-byte Folded Spill
	ldr	w9, [sp, #644]                  ; 4-byte Folded Reload
	str	w9, [sp, #892]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #640]                  ; 4-byte Folded Reload
	str	w9, [sp, #888]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #636]                  ; 4-byte Folded Reload
	str	w9, [sp, #884]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #632]                  ; 4-byte Folded Reload
	str	w9, [sp, #880]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #628]                  ; 4-byte Folded Reload
	str	w9, [sp, #876]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #624]                  ; 4-byte Folded Reload
	str	w9, [sp, #872]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #620]                  ; 4-byte Folded Reload
	str	w9, [sp, #868]                  ; 4-byte Folded Spill
	str	w16, [sp, #476]                 ; 4-byte Folded Spill
	mov	w9, #1
	b	LBB2_11
LBB2_10:                                ;   in Loop: Header=BB2_11 Depth=4
	ldr	w8, [sp, #700]                  ; 4-byte Folded Reload
	ldr	w16, [sp, #852]                 ; 4-byte Folded Reload
	add	w16, w16, w8
	ldr	w9, [sp, #868]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #868]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #872]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #872]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #876]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #876]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #880]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #880]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #884]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #884]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #888]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #888]                  ; 4-byte Folded Spill
	ldr	w9, [sp, #892]                  ; 4-byte Folded Reload
	add	w9, w9, w8
	str	w9, [sp, #892]                  ; 4-byte Folded Spill
	ldr	x9, [sp, #688]                  ; 8-byte Folded Reload
	ldr	x10, [sp, #896]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #896]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #904]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #904]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #912]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #912]                 ; 8-byte Folded Spill
	ldr	x10, [sp, #920]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #920]                 ; 8-byte Folded Spill
	ldr	x30, [sp, #856]                 ; 8-byte Folded Reload
	add	x30, x30, x9
	ldr	x10, [sp, #928]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	str	x10, [sp, #928]                 ; 8-byte Folded Spill
	ldr	w22, [sp, #864]                 ; 4-byte Folded Reload
	add	w22, w22, w8
	ldr	w15, [sp, #724]                 ; 4-byte Folded Reload
	add	w15, w15, w8
	ldr	x13, [sp, #728]                 ; 8-byte Folded Reload
	add	x13, x13, x9
	ldr	x6, [sp, #736]                  ; 8-byte Folded Reload
	add	x6, x6, x9
	ldr	w14, [sp, #744]                 ; 4-byte Folded Reload
	add	w14, w14, w8
	ldr	w27, [sp, #748]                 ; 4-byte Folded Reload
	add	w27, w27, w8
	ldr	x26, [sp, #752]                 ; 8-byte Folded Reload
	add	x26, x26, x9
	ldr	x24, [sp, #760]                 ; 8-byte Folded Reload
	add	x24, x24, x9
	ldr	w20, [sp, #768]                 ; 4-byte Folded Reload
	add	w20, w20, w8
	ldr	w5, [sp, #772]                  ; 4-byte Folded Reload
	add	w5, w5, w8
	ldr	x4, [sp, #776]                  ; 8-byte Folded Reload
	add	x4, x4, x9
	ldr	x3, [sp, #784]                  ; 8-byte Folded Reload
	add	x3, x3, x9
	ldr	w2, [sp, #792]                  ; 4-byte Folded Reload
	add	w2, w2, w8
	ldr	w1, [sp, #796]                  ; 4-byte Folded Reload
	add	w1, w1, w8
	ldr	x0, [sp, #800]                  ; 8-byte Folded Reload
	add	x0, x0, x9
	ldr	x17, [sp, #808]                 ; 8-byte Folded Reload
	add	x17, x17, x9
	ldr	x12, [sp, #816]                 ; 8-byte Folded Reload
	add	x12, x12, x9
	ldr	x11, [sp, #824]                 ; 8-byte Folded Reload
	add	x11, x11, x9
	ldr	x10, [sp, #832]                 ; 8-byte Folded Reload
	add	x10, x10, x9
	ldr	x8, [sp, #840]                  ; 8-byte Folded Reload
	add	x8, x8, x9
	ldr	x9, [sp, #712]                  ; 8-byte Folded Reload
	cmp	x9, x21
	b.hi	LBB2_8
LBB2_11:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_7 Depth=2
                                        ;       Parent Loop BB2_9 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB2_12 Depth 5
                                        ;             Child Loop BB2_13 Depth 6
	str	x30, [sp, #856]                 ; 8-byte Folded Spill
	str	w16, [sp, #852]                 ; 4-byte Folded Spill
	str	w22, [sp, #864]                 ; 4-byte Folded Spill
	mov	x22, x9
	add	x9, x9, #8
	str	x9, [sp, #712]                  ; 8-byte Folded Spill
	mov	x9, x6
	add	x6, x22, #1
	add	x16, x22, #2
	stur	x16, [x29, #-112]               ; 8-byte Folded Spill
	add	x16, x22, #3
	stur	x16, [x29, #-144]               ; 8-byte Folded Spill
	add	x16, x22, #4
	str	x16, [sp, #984]                 ; 8-byte Folded Spill
	add	x16, x22, #5
	str	x16, [sp, #968]                 ; 8-byte Folded Spill
	add	x16, x22, #6
	str	x16, [sp, #936]                 ; 8-byte Folded Spill
	str	x8, [sp, #840]                  ; 8-byte Folded Spill
	mov	x30, x13
	mov	x13, x8
	str	x10, [sp, #832]                 ; 8-byte Folded Spill
	mov	x8, x14
	mov	x14, x10
	str	x11, [sp, #824]                 ; 8-byte Folded Spill
	mov	x10, x15
	mov	x15, x11
	str	x12, [sp, #816]                 ; 8-byte Folded Spill
	mov	x16, x12
	str	x17, [sp, #808]                 ; 8-byte Folded Spill
	str	x0, [sp, #800]                  ; 8-byte Folded Spill
	str	w1, [sp, #796]                  ; 4-byte Folded Spill
	stur	w1, [x29, #-232]                ; 4-byte Folded Spill
	str	w2, [sp, #792]                  ; 4-byte Folded Spill
	stur	w2, [x29, #-212]                ; 4-byte Folded Spill
	str	x3, [sp, #784]                  ; 8-byte Folded Spill
	str	x4, [sp, #776]                  ; 8-byte Folded Spill
	str	w5, [sp, #772]                  ; 4-byte Folded Spill
	stur	w5, [x29, #-180]                ; 4-byte Folded Spill
	str	w20, [sp, #768]                 ; 4-byte Folded Spill
	stur	w20, [x29, #-148]               ; 4-byte Folded Spill
	str	x24, [sp, #760]                 ; 8-byte Folded Spill
	stur	x24, [x29, #-120]               ; 8-byte Folded Spill
	str	x26, [sp, #752]                 ; 8-byte Folded Spill
	mov	x24, x26
	str	w27, [sp, #748]                 ; 4-byte Folded Spill
	stur	w27, [x29, #-124]               ; 4-byte Folded Spill
	str	w8, [sp, #744]                  ; 4-byte Folded Spill
	mov	x12, x8
	str	x9, [sp, #736]                  ; 8-byte Folded Spill
	stur	x9, [x29, #-160]                ; 8-byte Folded Spill
	str	x30, [sp, #728]                 ; 8-byte Folded Spill
	stur	x30, [x29, #-136]               ; 8-byte Folded Spill
	str	w10, [sp, #724]                 ; 4-byte Folded Spill
	ldr	w9, [sp, #864]                  ; 4-byte Folded Reload
	ldr	x8, [sp, #928]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-192]                ; 8-byte Folded Spill
	ldr	x8, [sp, #856]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-176]                ; 8-byte Folded Spill
	ldr	x8, [sp, #920]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-224]                ; 8-byte Folded Spill
	ldr	x8, [sp, #912]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-208]                ; 8-byte Folded Spill
	ldr	x8, [sp, #904]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-240]                ; 8-byte Folded Spill
	ldr	x8, [sp, #896]                  ; 8-byte Folded Reload
	stur	x8, [x29, #-248]                ; 8-byte Folded Spill
	ldr	w27, [sp, #892]                 ; 4-byte Folded Reload
	ldr	w1, [sp, #888]                  ; 4-byte Folded Reload
	ldr	w2, [sp, #884]                  ; 4-byte Folded Reload
	ldr	w8, [sp, #880]                  ; 4-byte Folded Reload
	ldr	w11, [sp, #876]                 ; 4-byte Folded Reload
	stur	w11, [x29, #-164]               ; 4-byte Folded Spill
	ldr	w11, [sp, #872]                 ; 4-byte Folded Reload
	stur	w11, [x29, #-196]               ; 4-byte Folded Spill
	ldr	w11, [sp, #868]                 ; 4-byte Folded Reload
	stur	w11, [x29, #-228]               ; 4-byte Folded Spill
	ldr	w11, [sp, #852]                 ; 4-byte Folded Reload
	stur	w11, [x29, #-252]               ; 4-byte Folded Spill
	ldr	x11, [sp, #704]                 ; 8-byte Folded Reload
LBB2_12:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_7 Depth=2
                                        ;       Parent Loop BB2_9 Depth=3
                                        ;         Parent Loop BB2_11 Depth=4
                                        ; =>        This Loop Header: Depth=5
                                        ;             Child Loop BB2_13 Depth 6
	str	x11, [sp, #976]                 ; 8-byte Folded Spill
	mov	x30, #0
	mov	w26, #0
	ldr	x20, [sp, #960]                 ; 8-byte Folded Reload
LBB2_13:                                ;   Parent Loop BB2_5 Depth=1
                                        ;     Parent Loop BB2_7 Depth=2
                                        ;       Parent Loop BB2_9 Depth=3
                                        ;         Parent Loop BB2_11 Depth=4
                                        ;           Parent Loop BB2_12 Depth=5
                                        ; =>          This Inner Loop Header: Depth=6
	ldr	s0, [x16, x30]
	add	x11, x15, x30
	ldur	s1, [x11, #-8]
	ldr	s2, [x11]
	fadd	s1, s1, s2
	add	w5, w27, w26
	ldr	s2, [x19, w5, sxtw #2]
	add	w5, w9, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s1, s1, s2
	fadd	s1, s1, s3
	ldr	s2, [x13, x30]
	ldr	s3, [x14, x30]
	fadd	s1, s1, s2
	fadd	s1, s1, s3
	fmul	s1, s8, s1
	fmadd	s0, s0, s9, s1
	stur	s0, [x11, #-4]
	cmp	x22, x21
	b.hs	LBB2_21
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=6
	ldr	s1, [x17, x30]
	add	x11, x14, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	add	w5, w1, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	add	w5, w10, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldr	s3, [x0, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	cmp	x6, x21
	b.hs	LBB2_21
; %bb.15:                               ;   in Loop: Header=BB2_13 Depth=6
	ldr	s1, [x3, x30]
	add	x11, x0, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	add	w5, w2, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	add	w5, w12, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldr	s3, [x4, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	ldur	x11, [x29, #-112]               ; 8-byte Folded Reload
	cmp	x11, x21
	b.hs	LBB2_21
; %bb.16:                               ;   in Loop: Header=BB2_13 Depth=6
	ldur	x11, [x29, #-120]               ; 8-byte Folded Reload
	ldr	s1, [x11, x30]
	add	x11, x4, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	add	w5, w8, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-124]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldr	s3, [x24, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	ldur	x11, [x29, #-144]               ; 8-byte Folded Reload
	cmp	x11, x21
	b.hs	LBB2_21
; %bb.17:                               ;   in Loop: Header=BB2_13 Depth=6
	ldur	x11, [x29, #-160]               ; 8-byte Folded Reload
	ldr	s1, [x11, x30]
	add	x11, x24, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-164]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-148]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	x5, [x29, #-136]                ; 8-byte Folded Reload
	ldr	s3, [x5, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	ldr	x11, [sp, #984]                 ; 8-byte Folded Reload
	cmp	x11, x21
	b.hs	LBB2_21
; %bb.18:                               ;   in Loop: Header=BB2_13 Depth=6
	ldur	x11, [x29, #-192]               ; 8-byte Folded Reload
	ldr	s1, [x11, x30]
	ldur	x11, [x29, #-136]               ; 8-byte Folded Reload
	add	x11, x11, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-196]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-180]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	x5, [x29, #-176]                ; 8-byte Folded Reload
	ldr	s3, [x5, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	ldr	x11, [sp, #968]                 ; 8-byte Folded Reload
	cmp	x11, x21
	b.hs	LBB2_21
; %bb.19:                               ;   in Loop: Header=BB2_13 Depth=6
	ldur	x11, [x29, #-224]               ; 8-byte Folded Reload
	ldr	s1, [x11, x30]
	ldur	x11, [x29, #-176]               ; 8-byte Folded Reload
	add	x11, x11, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-228]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-212]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	x5, [x29, #-208]                ; 8-byte Folded Reload
	ldr	s3, [x5, x30]
	fadd	s0, s2, s0
	fadd	s0, s0, s3
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
	ldr	x11, [sp, #936]                 ; 8-byte Folded Reload
	cmp	x11, x21
	b.hs	LBB2_21
; %bb.20:                               ;   in Loop: Header=BB2_13 Depth=6
	ldur	x11, [x29, #-240]               ; 8-byte Folded Reload
	ldr	s1, [x11, x30]
	ldur	x11, [x29, #-208]               ; 8-byte Folded Reload
	add	x11, x11, x30
	ldur	s2, [x11, #-4]
	ldr	s3, [x11, #4]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-252]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	ldur	w5, [x29, #-232]                ; 4-byte Folded Reload
	add	w5, w5, w26
	ldr	s3, [x19, w5, sxtw #2]
	fadd	s2, s2, s3
	fadd	s0, s2, s0
	ldur	x5, [x29, #-248]                ; 8-byte Folded Reload
	ldr	s2, [x5, x30]
	fadd	s0, s0, s2
	fmul	s0, s8, s0
	fmadd	s0, s1, s9, s0
	str	s0, [x11]
LBB2_21:                                ;   in Loop: Header=BB2_13 Depth=6
	add	x11, x20, #1
	cmp	x11, x7
	b.hs	LBB2_23
; %bb.22:                               ;   in Loop: Header=BB2_13 Depth=6
	add	w26, w26, w25
	add	x30, x30, x28
	cmp	x20, x23
	mov	x20, x11
	b.lo	LBB2_13
LBB2_23:                                ;   in Loop: Header=BB2_12 Depth=5
	ldr	x20, [sp, #976]                 ; 8-byte Folded Reload
	add	x11, x20, #1
	ldr	x5, [sp, #952]                  ; 8-byte Folded Reload
	cmp	x11, x5
	b.hs	LBB2_10
; %bb.24:                               ;   in Loop: Header=BB2_12 Depth=5
	ldur	w5, [x29, #-252]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-252]                ; 4-byte Folded Spill
	ldur	w5, [x29, #-228]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-228]                ; 4-byte Folded Spill
	ldur	w5, [x29, #-196]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-196]                ; 4-byte Folded Spill
	ldur	w5, [x29, #-164]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-164]                ; 4-byte Folded Spill
	add	w8, w8, #1
	add	w2, w2, #1
	add	w1, w1, #1
	add	w27, w27, #1
	ldur	x5, [x29, #-248]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-248]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-240]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-240]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-208]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-208]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-224]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-224]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-176]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-176]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-192]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-192]                ; 8-byte Folded Spill
	add	w9, w9, #1
	add	w10, w10, #1
	ldur	x5, [x29, #-136]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-136]                ; 8-byte Folded Spill
	ldur	x5, [x29, #-160]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-160]                ; 8-byte Folded Spill
	add	w12, w12, #1
	ldur	w5, [x29, #-124]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-124]                ; 4-byte Folded Spill
	add	x24, x24, #4
	ldur	x5, [x29, #-120]                ; 8-byte Folded Reload
	add	x5, x5, #4
	stur	x5, [x29, #-120]                ; 8-byte Folded Spill
	ldur	w5, [x29, #-148]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-148]                ; 4-byte Folded Spill
	ldur	w5, [x29, #-180]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-180]                ; 4-byte Folded Spill
	add	x4, x4, #4
	add	x3, x3, #4
	ldur	w5, [x29, #-212]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-212]                ; 4-byte Folded Spill
	ldur	w5, [x29, #-232]                ; 4-byte Folded Reload
	add	w5, w5, #1
	stur	w5, [x29, #-232]                ; 4-byte Folded Spill
	add	x0, x0, #4
	add	x17, x17, #4
	add	x16, x16, #4
	add	x15, x15, #4
	add	x14, x14, #4
	add	x13, x13, #4
	ldr	x5, [sp, #944]                  ; 8-byte Folded Reload
	cmp	x20, x5
	b.lo	LBB2_12
	b	LBB2_10
LBB2_25:
	ldp	w23, w21, [sp, #236]            ; 8-byte Folded Reload
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	ldr	w20, [sp, #232]                 ; 4-byte Folded Reload
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	bl	__Z7set_bndiiiiPf
	mov	x0, x23
	mov	x1, x22
	mov	x2, x21
	mov	x3, x20
	mov	x4, x19
	add	sp, sp, #1152
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #112              ; 16-byte Folded Reload
	b	__Z7set_bndiiiiPf
LBB2_26:
	add	sp, sp, #1152
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #112              ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__Z7diffuseiiiiPfS_ff           ; -- Begin function _Z7diffuseiiiiPfS_ff
	.p2align	2
__Z7diffuseiiiiPfS_ff:                  ; @_Z7diffuseiiiiPfS_ff
	.cfi_startproc
; %bb.0:
	cmp	w0, w1
	csel	w8, w0, w1, gt
	cmp	w8, w2
	csel	w8, w8, w2, gt
	fmul	s0, s0, s1
	scvtf	s1, w8
	fmul	s0, s0, s1
	fmul	s0, s0, s1
	fmov	s1, #1.00000000
	fmov	s2, #6.00000000
	fmadd	s1, s0, s2, s1
	b	__Z9lin_solveiiiiPfS_ff
	.cfi_endproc
                                        ; -- End function
	.globl	__Z6advectiiiiPfS_S_S_S_f       ; -- Begin function _Z6advectiiiiPfS_S_S_S_f
	.p2align	2
__Z6advectiiiiPfS_S_S_S_f:              ; @_Z6advectiiiiPfS_S_S_S_f
	.cfi_startproc
; %bb.0:
	stp	x28, x27, [sp, #-80]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x26, x25, [sp, #16]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w27, -72
	.cfi_offset w28, -80
	cmp	w0, #1
	b.lt	LBB4_9
; %bb.1:
	cmp	w1, #1
	b.lt	LBB4_9
; %bb.2:
	cmp	w2, #1
	b.lt	LBB4_9
; %bb.3:
	mov	x8, #0
	scvtf	s5, w2
	scvtf	s4, w1
	scvtf	s3, w0
	ldr	x19, [sp, #80]
	add	w9, w0, #2
	add	w10, w1, #2
	mul	w10, w10, w9
	fneg	s1, s3
	fmul	s1, s1, s0
	fneg	s2, s4
	fmul	s2, s2, s0
	fneg	s6, s5
	fmul	s0, s6, s0
	fmov	s6, #0.50000000
	fadd	s3, s3, s6
	fadd	s4, s4, s6
	sxtw	x14, w10
	add	w11, w2, #1
	add	w12, w1, #1
	add	w13, w0, #1
	add	x14, x14, x9
	lsl	x14, x14, #2
	fadd	s5, s5, s6
	add	x20, x14, #4
	add	x14, x6, x20
	ubfiz	x15, x9, #2, #32
	sbfiz	x16, x10, #2, #32
	add	x17, x7, x20
	add	x6, x19, x20
	add	x7, x4, x20
	mov	w19, #1
	fmov	s6, #0.50000000
	fmov	s7, #1.00000000
LBB4_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_5 Depth 2
                                        ;       Child Loop BB4_6 Depth 3
	scvtf	s16, w19
	mov	x20, x8
	mov	w21, #1
LBB4_5:                                 ;   Parent Loop BB4_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB4_6 Depth 3
	scvtf	s17, w21
	mov	x22, x20
	mov	w23, #1
LBB4_6:                                 ;   Parent Loop BB4_4 Depth=1
                                        ;     Parent Loop BB4_5 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	s18, [x14, x22]
	fmadd	s18, s1, s18, s16
	ldr	s19, [x17, x22]
	fmadd	s19, s2, s19, s17
	scvtf	s20, w23
	ldr	s21, [x6, x22]
	fmadd	s20, s0, s21, s20
	fmax	s18, s18, s6
	fcmp	s18, s3
	fcsel	s18, s3, s18, gt
	fmax	s19, s19, s6
	fcmp	s19, s4
	fcsel	s19, s4, s19, gt
	fmax	s20, s20, s6
	fcmp	s20, s5
	fcvtzs	w24, s18
	fcvtzs	w25, s19
	fcsel	s20, s5, s20, gt
	fcvtzs	w26, s20
	fcvtzs	s21, s18
	scvtf	s21, s21
	fsub	s18, s18, s21
	fcvtzs	s21, s19
	scvtf	s21, s21
	fsub	s19, s19, s21
	fcvtzs	s21, s20
	scvtf	s21, s21
	fsub	s20, s20, s21
	fsub	s21, s7, s20
	madd	w24, w9, w25, w24
	mul	w25, w10, w26
	add	w26, w24, w25
	add	x26, x5, w26, sxtw #2
	add	w27, w10, w25
	add	w28, w27, w24
	add	x28, x5, w28, sxtw #2
	ldp	s22, s23, [x28]
	fmul	s22, s22, s20
	ldp	s24, s25, [x26]
	fmadd	s22, s21, s24, s22
	add	w24, w24, w9
	add	w25, w24, w25
	add	x25, x5, w25, sxtw #2
	add	w24, w24, w27
	add	x24, x5, w24, sxtw #2
	ldp	s24, s26, [x24]
	fmul	s24, s24, s20
	ldp	s27, s28, [x25]
	fmadd	s24, s21, s27, s24
	fsub	s27, s7, s19
	fmul	s24, s19, s24
	fmadd	s22, s27, s22, s24
	fmul	s23, s23, s20
	fmadd	s23, s21, s25, s23
	fsub	s24, s7, s18
	fmul	s20, s20, s26
	fmadd	s20, s21, s28, s20
	fmul	s19, s19, s20
	fmadd	s19, s27, s23, s19
	fmul	s18, s18, s19
	fmadd	s18, s24, s22, s18
	str	s18, [x7, x22]
	add	x23, x23, #1
	add	x22, x22, x16
	cmp	x11, x23
	b.ne	LBB4_6
; %bb.7:                                ;   in Loop: Header=BB4_5 Depth=2
	add	x21, x21, #1
	add	x20, x20, x15
	cmp	x21, x12
	b.ne	LBB4_5
; %bb.8:                                ;   in Loop: Header=BB4_4 Depth=1
	add	x19, x19, #1
	add	x8, x8, #4
	cmp	x19, x13
	b.ne	LBB4_4
LBB4_9:
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #16]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp], #80             ; 16-byte Folded Reload
	b	__Z7set_bndiiiiPf
	.cfi_endproc
                                        ; -- End function
	.globl	__Z7projectiiiPfS_S_S_S_        ; -- Begin function _Z7projectiiiPfS_S_S_S_
	.p2align	2
__Z7projectiiiPfS_S_S_S_:               ; @_Z7projectiiiPfS_S_S_S_
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	x28, x27, [sp, #32]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #112]            ; 16-byte Folded Spill
	add	x29, sp, #112
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	x20, x7
	mov	x25, x6
	mov	x23, x4
	mov	x24, x3
	mov	x19, x2
	mov	x21, x1
	mov	x22, x0
	cmp	w0, #1
	str	x5, [sp, #16]                   ; 8-byte Folded Spill
	str	w2, [sp, #28]                   ; 4-byte Folded Spill
	str	w1, [sp, #12]                   ; 4-byte Folded Spill
	b.lt	LBB5_15
; %bb.1:
	cmp	w21, w19
	csel	w8, w21, w19, gt
	cmp	w8, w22
	csel	w10, w8, w22, gt
	cmp	w21, #1
	b.lt	LBB5_15
; %bb.2:
	cmp	w19, #1
	b.lt	LBB5_15
; %bb.3:
	mov	x8, #0
	mov	w9, #0
	add	w27, w22, #2
	scvtf	s0, w10
	add	w10, w21, #2
	mul	w17, w10, w27
	sxtw	x28, w17
	add	w0, w19, #1
	add	w19, w21, #1
	add	w26, w22, #1
	add	w10, w17, w22, lsl #1
	add	w10, w10, #5
	add	w11, w22, w17
	add	x12, x28, x27
	lsl	x12, x12, #2
	add	x15, x12, #4
	str	x20, [sp]                       ; 8-byte Folded Spill
	add	x12, x20, x15
	ubfiz	x13, x27, #2, #32
	sbfiz	x14, x17, #2, #32
	add	x15, x25, x15
	add	x16, x13, x5
	add	x16, x16, #4
	sub	x20, x0, #1
	add	x17, x16, w17, sxtw #3
	mov	w0, #1
	fmov	s1, #-0.50000000
LBB5_4:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_5 Depth 2
                                        ;       Child Loop BB5_6 Depth 3
	mov	x1, x8
	mov	x2, x9
	mov	w3, #1
LBB5_5:                                 ;   Parent Loop BB5_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_6 Depth 3
	add	x3, x3, #1
	mov	x4, x20
	mov	x5, x1
	mov	x6, x2
LBB5_6:                                 ;   Parent Loop BB5_4 Depth=1
                                        ;     Parent Loop BB5_5 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	add	w7, w11, w6
	add	w30, w7, #4
	ldr	s2, [x24, w30, sxtw #2]
	add	w7, w7, #2
	ldr	s3, [x24, w7, sxtw #2]
	fsub	s2, s2, s3
	add	w7, w10, w6
	ldr	s3, [x23, w7, sxtw #2]
	fadd	s2, s2, s3
	add	w6, w28, w6
	add	w7, w6, #1
	ldr	s3, [x23, w7, sxtw #2]
	fsub	s2, s2, s3
	ldr	s3, [x17, x5]
	fadd	s2, s2, s3
	ldr	s3, [x16, x5]
	fsub	s2, s2, s3
	fmul	s2, s2, s1
	fdiv	s2, s2, s0
	str	s2, [x12, x5]
	str	wzr, [x15, x5]
	add	x5, x5, x14
	subs	x4, x4, #1
	b.ne	LBB5_6
; %bb.7:                                ;   in Loop: Header=BB5_5 Depth=2
	add	w2, w2, w27
	add	x1, x1, x13
	cmp	x3, x19
	b.ne	LBB5_5
; %bb.8:                                ;   in Loop: Header=BB5_4 Depth=1
	add	x0, x0, #1
	add	w9, w9, #1
	add	x8, x8, #4
	cmp	x0, x26
	b.ne	LBB5_4
; %bb.9:
	mov	x0, x22
	mov	x1, x21
	ldr	w2, [sp, #28]                   ; 4-byte Folded Reload
	mov	w3, #0
	ldr	x4, [sp]                        ; 8-byte Folded Reload
	bl	__Z7set_bndiiiiPf
	mov	x0, x22
	mov	x1, x21
	ldr	w21, [sp, #28]                  ; 4-byte Folded Reload
	mov	x2, x21
	mov	w3, #0
	mov	x4, x25
	bl	__Z7set_bndiiiiPf
	fmov	s0, #1.00000000
	fmov	s1, #6.00000000
	mov	x0, x22
	ldr	w1, [sp, #12]                   ; 4-byte Folded Reload
	mov	x2, x21
	mov	w3, #0
	mov	x4, x25
	ldr	x5, [sp]                        ; 8-byte Folded Reload
	bl	__Z9lin_solveiiiiPfS_ff
	mov	x8, #0
	mov	w9, #0
	add	w10, w28, w22, lsl #1
	add	w10, w10, #5
	add	w11, w22, w28
	add	x12, x28, x27
	lsl	x12, x12, #2
	add	x16, x12, #4
	add	x12, x24, x16
	lsl	x13, x27, #2
	lsl	x14, x28, #2
	add	x15, x23, x16
	ldr	x17, [sp, #16]                  ; 8-byte Folded Reload
	add	x16, x17, x16
	add	x17, x13, x25
	add	x17, x17, #4
	fmov	s0, #-0.50000000
	mov	w0, #1
	add	x1, x17, x28, lsl #3
LBB5_10:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_11 Depth 2
                                        ;       Child Loop BB5_12 Depth 3
	mov	x2, x8
	mov	x3, x9
	mov	w4, #1
LBB5_11:                                ;   Parent Loop BB5_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_12 Depth 3
	add	x4, x4, #1
	mov	x5, x20
	mov	x6, x2
	mov	x7, x3
LBB5_12:                                ;   Parent Loop BB5_10 Depth=1
                                        ;     Parent Loop BB5_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	add	w30, w11, w7
	add	w21, w30, #4
	ldr	s1, [x25, w21, sxtw #2]
	add	w21, w30, #2
	ldr	s2, [x25, w21, sxtw #2]
	fsub	s1, s1, s2
	ldr	s2, [x12, x6]
	fmadd	s1, s1, s0, s2
	str	s1, [x12, x6]
	add	w21, w10, w7
	ldr	s1, [x25, w21, sxtw #2]
	add	w7, w28, w7
	add	w21, w7, #1
	ldr	s2, [x25, w21, sxtw #2]
	fsub	s1, s1, s2
	ldr	s2, [x15, x6]
	fmadd	s1, s1, s0, s2
	str	s1, [x15, x6]
	ldr	s1, [x1, x6]
	ldr	s2, [x17, x6]
	ldr	s3, [x16, x6]
	fsub	s1, s1, s2
	fmadd	s1, s1, s0, s3
	str	s1, [x16, x6]
	add	x6, x6, x14
	subs	x5, x5, #1
	b.ne	LBB5_12
; %bb.13:                               ;   in Loop: Header=BB5_11 Depth=2
	add	w3, w3, w27
	add	x2, x2, x13
	cmp	x4, x19
	b.ne	LBB5_11
; %bb.14:                               ;   in Loop: Header=BB5_10 Depth=1
	add	x0, x0, #1
	add	w9, w9, #1
	add	x8, x8, #4
	cmp	x0, x26
	b.ne	LBB5_10
	b	LBB5_16
LBB5_15:
	mov	x0, x22
	mov	x1, x21
	mov	x2, x19
	mov	w3, #0
	mov	x4, x20
	bl	__Z7set_bndiiiiPf
	mov	x0, x22
	mov	x1, x21
	mov	x2, x19
	mov	w3, #0
	mov	x4, x25
	bl	__Z7set_bndiiiiPf
	fmov	s0, #1.00000000
	fmov	s1, #6.00000000
	mov	x0, x22
	mov	x1, x21
	mov	x2, x19
	mov	w3, #0
	mov	x4, x25
	mov	x5, x20
	bl	__Z9lin_solveiiiiPfS_ff
LBB5_16:
	mov	x0, x22
	ldr	w20, [sp, #12]                  ; 4-byte Folded Reload
	mov	x1, x20
	ldr	w19, [sp, #28]                  ; 4-byte Folded Reload
	mov	x2, x19
	mov	w3, #1
	mov	x4, x24
	bl	__Z7set_bndiiiiPf
	mov	x0, x22
	mov	x1, x20
	mov	x2, x19
	mov	w3, #2
	mov	x4, x23
	bl	__Z7set_bndiiiiPf
	mov	x0, x22
	mov	x1, x20
	mov	x2, x19
	mov	w3, #3
	ldr	x4, [sp, #16]                   ; 8-byte Folded Reload
	ldp	x29, x30, [sp, #112]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #96]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #128
	b	__Z7set_bndiiiiPf
	.cfi_endproc
                                        ; -- End function
	.globl	__Z9dens_stepiiiPfS_S_S_S_ff    ; -- Begin function _Z9dens_stepiiiPfS_S_S_S_ff
	.p2align	2
__Z9dens_stepiiiPfS_S_S_S_ff:           ; @_Z9dens_stepiiiPfS_S_S_S_ff
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #112
	.cfi_def_cfa_offset 112
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	mov	x24, x7
	mov	x19, x6
	mov	x20, x5
	mov	x21, x4
	mov	x22, x3
	mov	x23, x2
	fmov	s18, s1
	mov	x25, x1
	mov	x26, x0
	add	w8, w0, #2
	add	w9, w1, #2
	mul	w8, w9, w8
	add	w9, w2, #2
	mul	w8, w8, w9
	cmp	w8, #1
	b.lt	LBB6_9
; %bb.1:
	cmp	w8, #16
	b.lo	LBB6_6
; %bb.2:
	lsl	x9, x8, #2
	add	x10, x22, x9
	add	x9, x21, x9
	cmp	x9, x22
	ccmp	x10, x21, #0, hi
	b.hi	LBB6_6
; %bb.3:
	and	x9, x8, #0xfffffff0
	dup.4s	v1, v18[0]
	add	x10, x22, #32
	add	x11, x21, #32
	mov	x12, x9
LBB6_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x11, #-32]
	ldp	q4, q5, [x11], #64
	ldp	q6, q7, [x10, #-32]
	ldp	q16, q17, [x10]
	fmla.4s	v6, v2, v1
	fmla.4s	v7, v3, v1
	fmla.4s	v16, v4, v1
	fmla.4s	v17, v5, v1
	stp	q6, q7, [x10, #-32]
	stp	q16, q17, [x10], #64
	subs	x12, x12, #16
	b.ne	LBB6_4
; %bb.5:
	cmp	x9, x8
	b.ne	LBB6_7
	b	LBB6_9
LBB6_6:
	mov	x9, #0
LBB6_7:
	sub	x8, x8, x9
	lsl	x10, x9, #2
	add	x9, x22, x10
	add	x10, x21, x10
LBB6_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x10], #4
	ldr	s2, [x9]
	fmadd	s1, s18, s1, s2
	str	s1, [x9], #4
	subs	x8, x8, #1
	b.ne	LBB6_8
LBB6_9:
	cmp	w26, w25
	csel	w8, w26, w25, gt
	cmp	w8, w23
	csel	w8, w8, w23, gt
	scvtf	s1, w8
	fmul	s0, s0, s18
	fmul	s0, s0, s1
	fmul	s0, s0, s1
	fmov	s1, #1.00000000
	fmov	s2, #6.00000000
	fmadd	s1, s0, s2, s1
	mov	x0, x26
	mov	x1, x25
	mov	x2, x23
	mov	w3, #0
	mov	x4, x21
	mov	x5, x22
	str	q18, [sp, #16]                  ; 16-byte Folded Spill
	bl	__Z9lin_solveiiiiPfS_ff
	str	x24, [sp]
	mov	x0, x26
	mov	x1, x25
	mov	x2, x23
	mov	w3, #0
	mov	x4, x22
	mov	x5, x21
	mov	x6, x20
	mov	x7, x19
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $s0 killed $s0 killed $q0
	bl	__Z6advectiiiiPfS_S_S_S_f
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #112
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__Z8vel_stepiiiPfS_S_S_S_S_ff   ; -- Begin function _Z8vel_stepiiiPfS_S_S_S_S_ff
	.p2align	2
__Z8vel_stepiiiPfS_S_S_S_S_ff:          ; @_Z8vel_stepiiiPfS_S_S_S_S_ff
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #144
	.cfi_def_cfa_offset 144
	stp	d9, d8, [sp, #32]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #48]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #64]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #80]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #96]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #128]            ; 16-byte Folded Spill
	add	x29, sp, #128
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x19, x7
	mov	x20, x6
	mov	x21, x5
	mov	x22, x4
	mov	x23, x3
	mov	x24, x2
	fmov	s18, s1
	mov	x25, x1
	mov	x26, x0
	ldr	x27, [x29, #16]
	add	w8, w0, #2
	add	w9, w1, #2
	mul	w8, w9, w8
	add	w9, w2, #2
	mul	w8, w8, w9
	cmp	w8, #1
	b.lt	LBB7_21
; %bb.1:
	lsl	x9, x8, #2
	cmp	w8, #16
	b.lo	LBB7_6
; %bb.2:
	add	x10, x23, x9
	add	x11, x20, x9
	cmp	x11, x23
	ccmp	x10, x20, #0, hi
	b.hi	LBB7_6
; %bb.3:
	and	x10, x8, #0xfffffff0
	dup.4s	v1, v18[0]
	add	x11, x23, #32
	add	x12, x20, #32
	mov	x13, x10
LBB7_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x12, #-32]
	ldp	q4, q5, [x12], #64
	ldp	q6, q7, [x11, #-32]
	ldp	q16, q17, [x11]
	fmla.4s	v6, v2, v1
	fmla.4s	v7, v3, v1
	fmla.4s	v16, v4, v1
	fmla.4s	v17, v5, v1
	stp	q6, q7, [x11, #-32]
	stp	q16, q17, [x11], #64
	subs	x13, x13, #16
	b.ne	LBB7_4
; %bb.5:
	cmp	x10, x8
	b.ne	LBB7_7
	b	LBB7_9
LBB7_6:
	mov	x10, #0
LBB7_7:
	sub	x11, x8, x10
	lsl	x12, x10, #2
	add	x10, x23, x12
	add	x12, x20, x12
LBB7_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x12], #4
	ldr	s2, [x10]
	fmadd	s1, s18, s1, s2
	str	s1, [x10], #4
	subs	x11, x11, #1
	b.ne	LBB7_8
LBB7_9:
	cmp	w8, #16
	b.lo	LBB7_12
; %bb.10:
	add	x10, x19, x9
	cmp	x10, x22
	b.ls	LBB7_22
; %bb.11:
	add	x10, x22, x9
	cmp	x10, x19
	b.ls	LBB7_22
LBB7_12:
	mov	x10, #0
LBB7_13:
	sub	x11, x8, x10
	lsl	x12, x10, #2
	add	x10, x22, x12
	add	x12, x19, x12
LBB7_14:                                ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x12], #4
	ldr	s2, [x10]
	fmadd	s1, s18, s1, s2
	str	s1, [x10], #4
	subs	x11, x11, #1
	b.ne	LBB7_14
LBB7_15:
	cmp	w8, #16
	b.lo	LBB7_18
; %bb.16:
	add	x10, x27, x9
	cmp	x10, x21
	b.ls	LBB7_25
; %bb.17:
	add	x9, x21, x9
	cmp	x9, x27
	b.ls	LBB7_25
LBB7_18:
	mov	x9, #0
LBB7_19:
	sub	x8, x8, x9
	lsl	x10, x9, #2
	add	x9, x21, x10
	add	x10, x27, x10
LBB7_20:                                ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x10], #4
	ldr	s2, [x9]
	fmadd	s1, s18, s1, s2
	str	s1, [x9], #4
	subs	x8, x8, #1
	b.ne	LBB7_20
LBB7_21:
	cmp	w26, w25
	csel	w8, w26, w25, gt
	cmp	w8, w24
	csel	w8, w8, w24, gt
	scvtf	s1, w8
	fmul	s0, s0, s18
	fmul	s0, s0, s1
	fmul	s8, s0, s1
	fmov	s0, #1.00000000
	fmov	s1, #6.00000000
	fmadd	s9, s8, s1, s0
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #1
	mov	x4, x20
	mov	x5, x23
	fmov	s0, s8
	fmov	s1, s9
	str	q18, [sp, #16]                  ; 16-byte Folded Spill
	bl	__Z9lin_solveiiiiPfS_ff
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #2
	mov	x4, x19
	mov	x5, x22
	fmov	s0, s8
	fmov	s1, s9
	bl	__Z9lin_solveiiiiPfS_ff
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #3
	mov	x4, x27
	mov	x5, x21
	fmov	s0, s8
	fmov	s1, s9
	bl	__Z9lin_solveiiiiPfS_ff
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	x3, x20
	mov	x4, x19
	mov	x5, x27
	mov	x6, x23
	mov	x7, x22
	bl	__Z7projectiiiPfS_S_S_S_
	str	x27, [sp]
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #1
	mov	x4, x23
	mov	x5, x20
	mov	x6, x20
	mov	x7, x19
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $s0 killed $s0 killed $q0
	bl	__Z6advectiiiiPfS_S_S_S_f
	str	x27, [sp]
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #2
	mov	x4, x22
	mov	x5, x19
	mov	x6, x20
	mov	x7, x19
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $s0 killed $s0 killed $q0
	bl	__Z6advectiiiiPfS_S_S_S_f
	str	x27, [sp]
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	w3, #3
	mov	x4, x21
	mov	x5, x27
	mov	x6, x20
	mov	x7, x19
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $s0 killed $s0 killed $q0
	bl	__Z6advectiiiiPfS_S_S_S_f
	mov	x0, x26
	mov	x1, x25
	mov	x2, x24
	mov	x3, x23
	mov	x4, x22
	mov	x5, x21
	mov	x6, x20
	mov	x7, x19
	ldp	x29, x30, [sp, #128]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #112]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #32]               ; 16-byte Folded Reload
	add	sp, sp, #144
	b	__Z7projectiiiPfS_S_S_S_
LBB7_22:
	and	x10, x8, #0xfffffff0
	dup.4s	v1, v18[0]
	add	x11, x22, #32
	add	x12, x19, #32
	mov	x13, x10
LBB7_23:                                ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x12, #-32]
	ldp	q4, q5, [x12], #64
	ldp	q6, q7, [x11, #-32]
	ldp	q16, q17, [x11]
	fmla.4s	v6, v2, v1
	fmla.4s	v7, v3, v1
	fmla.4s	v16, v4, v1
	fmla.4s	v17, v5, v1
	stp	q6, q7, [x11, #-32]
	stp	q16, q17, [x11], #64
	subs	x13, x13, #16
	b.ne	LBB7_23
; %bb.24:
	cmp	x10, x8
	b.eq	LBB7_15
	b	LBB7_13
LBB7_25:
	and	x9, x8, #0xfffffff0
	dup.4s	v1, v18[0]
	add	x10, x21, #32
	add	x11, x27, #32
	mov	x12, x9
LBB7_26:                                ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x11, #-32]
	ldp	q4, q5, [x11], #64
	ldp	q6, q7, [x10, #-32]
	ldp	q16, q17, [x10]
	fmla.4s	v6, v2, v1
	fmla.4s	v7, v3, v1
	fmla.4s	v16, v4, v1
	fmla.4s	v17, v5, v1
	stp	q6, q7, [x10, #-32]
	stp	q16, q17, [x10], #64
	subs	x12, x12, #16
	b.ne	LBB7_26
; %bb.27:
	cmp	x9, x8
	b.eq	LBB7_21
	b	LBB7_19
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
