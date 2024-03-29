/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) mobile_robot_cost_y_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_sq CASADI_PREFIX(sq)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[13] = {6, 6, 0, 0, 0, 2, 4, 4, 4, 2, 3, 2, 3};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s5[12] = {2, 3, 0, 2, 4, 6, 0, 1, 0, 1, 0, 1};

/* mobile_robot_cost_y_hess:(i0[4],i1[2],i2[],i3[7],i4[],i5[2x3])->(o0[6x6,4nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real *w0=w+0, w3, w4, *w5=w+6, w6, w7, w8, w9, *w10=w+17, w11, w12, w13, w14, w15, w16, *w17=w+29, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, w35, *w38=w+52, *w39=w+54, *w40=w+56;
  /* #0: @0 = zeros(6x6,4nz) */
  casadi_clear(w0, 4);
  /* #1: @1 = 00 */
  /* #2: @2 = 00 */
  /* #3: @3 = 6.25 */
  w3 = 6.2499999999999991e+00;
  /* #4: @4 = 25 */
  w4 = 25.;
  /* #5: @5 = input[3][0] */
  casadi_copy(arg[3], 7, w5);
  /* #6: {NULL, NULL, NULL, NULL, NULL, NULL, @6} = vertsplit(@5) */
  w6 = w5[6];
  /* #7: @7 = 1 */
  w7 = 1.;
  /* #8: @8 = 25 */
  w8 = 25.;
  /* #9: @9 = input[0][0] */
  w9 = arg[0] ? arg[0][0] : 0;
  /* #10: @10 = input[5][0] */
  casadi_copy(arg[5], 6, w10);
  /* #11: @11 = @10[4] */
  for (rr=(&w11), ss=w10+4; ss!=w10+5; ss+=1) *rr++ = *ss;
  /* #12: @11 = (@9-@11) */
  w11  = (w9-w11);
  /* #13: @12 = sq(@11) */
  w12 = casadi_sq( w11 );
  /* #14: @13 = 0.16 */
  w13 = 1.6000000000000003e-01;
  /* #15: @12 = (@12/@13) */
  w12 /= w13;
  /* #16: @13 = input[0][1] */
  w13 = arg[0] ? arg[0][1] : 0;
  /* #17: @14 = @10[5] */
  for (rr=(&w14), ss=w10+5; ss!=w10+6; ss+=1) *rr++ = *ss;
  /* #18: @14 = (@13-@14) */
  w14  = (w13-w14);
  /* #19: @15 = sq(@14) */
  w15 = casadi_sq( w14 );
  /* #20: @16 = 0.16 */
  w16 = 1.6000000000000003e-01;
  /* #21: @15 = (@15/@16) */
  w15 /= w16;
  /* #22: @12 = (@12+@15) */
  w12 += w15;
  /* #23: @12 = (@4*@12) */
  w12  = (w4*w12);
  /* #24: @8 = (@8-@12) */
  w8 -= w12;
  /* #25: @12 = sq(@8) */
  w12 = casadi_sq( w8 );
  /* #26: @7 = (@7+@12) */
  w7 += w12;
  /* #27: @12 = (@6/@7) */
  w12  = (w6/w7);
  /* #28: @15 = (@4*@12) */
  w15  = (w4*w12);
  /* #29: @16 = (@3*@15) */
  w16  = (w3*w15);
  /* #30: @17 = ones(6x1,5nz) */
  casadi_fill(w17, 5, 1.);
  /* #31: {NULL, NULL, @18, NULL, NULL, NULL} = vertsplit(@17) */
  w18 = w17[2];
  /* #32: @19 = (2.*@18) */
  w19 = (2.* w18 );
  /* #33: @16 = (@16*@19) */
  w16 *= w19;
  /* #34: @19 = (2.*@11) */
  w19 = (2.* w11 );
  /* #35: @12 = (@12/@7) */
  w12 /= w7;
  /* #36: @8 = (2.*@8) */
  w8 = (2.* w8 );
  /* #37: @7 = 6.25 */
  w7 = 6.2499999999999991e+00;
  /* #38: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #39: @11 = (@11*@18) */
  w11 *= w18;
  /* #40: @7 = (@7*@11) */
  w7 *= w11;
  /* #41: @7 = (@4*@7) */
  w7  = (w4*w7);
  /* #42: @7 = (@8*@7) */
  w7  = (w8*w7);
  /* #43: @7 = (@12*@7) */
  w7  = (w12*w7);
  /* #44: @7 = (@4*@7) */
  w7  = (w4*w7);
  /* #45: @11 = (@3*@7) */
  w11  = (w3*w7);
  /* #46: @11 = (@19*@11) */
  w11  = (w19*w11);
  /* #47: @16 = (@16+@11) */
  w16 += w11;
  /* #48: @16 = (-@16) */
  w16 = (- w16 );
  /* #49: @11 = 6.25 */
  w11 = 6.2499999999999991e+00;
  /* #50: @20 = 25 */
  w20 = 25.;
  /* #51: @21 = 1 */
  w21 = 1.;
  /* #52: @22 = 25 */
  w22 = 25.;
  /* #53: @23 = @10[2] */
  for (rr=(&w23), ss=w10+2; ss!=w10+3; ss+=1) *rr++ = *ss;
  /* #54: @23 = (@9-@23) */
  w23  = (w9-w23);
  /* #55: @24 = sq(@23) */
  w24 = casadi_sq( w23 );
  /* #56: @25 = 0.16 */
  w25 = 1.6000000000000003e-01;
  /* #57: @24 = (@24/@25) */
  w24 /= w25;
  /* #58: @25 = @10[3] */
  for (rr=(&w25), ss=w10+3; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #59: @25 = (@13-@25) */
  w25  = (w13-w25);
  /* #60: @26 = sq(@25) */
  w26 = casadi_sq( w25 );
  /* #61: @27 = 0.16 */
  w27 = 1.6000000000000003e-01;
  /* #62: @26 = (@26/@27) */
  w26 /= w27;
  /* #63: @24 = (@24+@26) */
  w24 += w26;
  /* #64: @24 = (@20*@24) */
  w24  = (w20*w24);
  /* #65: @22 = (@22-@24) */
  w22 -= w24;
  /* #66: @24 = sq(@22) */
  w24 = casadi_sq( w22 );
  /* #67: @21 = (@21+@24) */
  w21 += w24;
  /* #68: @24 = (@6/@21) */
  w24  = (w6/w21);
  /* #69: @26 = (@20*@24) */
  w26  = (w20*w24);
  /* #70: @27 = (@11*@26) */
  w27  = (w11*w26);
  /* #71: @28 = (2.*@18) */
  w28 = (2.* w18 );
  /* #72: @27 = (@27*@28) */
  w27 *= w28;
  /* #73: @28 = (2.*@23) */
  w28 = (2.* w23 );
  /* #74: @24 = (@24/@21) */
  w24 /= w21;
  /* #75: @22 = (2.*@22) */
  w22 = (2.* w22 );
  /* #76: @21 = 6.25 */
  w21 = 6.2499999999999991e+00;
  /* #77: @23 = (2.*@23) */
  w23 = (2.* w23 );
  /* #78: @23 = (@23*@18) */
  w23 *= w18;
  /* #79: @21 = (@21*@23) */
  w21 *= w23;
  /* #80: @21 = (@20*@21) */
  w21  = (w20*w21);
  /* #81: @21 = (@22*@21) */
  w21  = (w22*w21);
  /* #82: @21 = (@24*@21) */
  w21  = (w24*w21);
  /* #83: @21 = (@20*@21) */
  w21  = (w20*w21);
  /* #84: @23 = (@11*@21) */
  w23  = (w11*w21);
  /* #85: @23 = (@28*@23) */
  w23  = (w28*w23);
  /* #86: @27 = (@27+@23) */
  w27 += w23;
  /* #87: @16 = (@16-@27) */
  w16 -= w27;
  /* #88: @27 = 6.25 */
  w27 = 6.2499999999999991e+00;
  /* #89: @23 = 25 */
  w23 = 25.;
  /* #90: @29 = 1 */
  w29 = 1.;
  /* #91: @30 = 25 */
  w30 = 25.;
  /* #92: @31 = @10[0] */
  for (rr=(&w31), ss=w10+0; ss!=w10+1; ss+=1) *rr++ = *ss;
  /* #93: @9 = (@9-@31) */
  w9 -= w31;
  /* #94: @31 = sq(@9) */
  w31 = casadi_sq( w9 );
  /* #95: @32 = 0.16 */
  w32 = 1.6000000000000003e-01;
  /* #96: @31 = (@31/@32) */
  w31 /= w32;
  /* #97: @32 = @10[1] */
  for (rr=(&w32), ss=w10+1; ss!=w10+2; ss+=1) *rr++ = *ss;
  /* #98: @13 = (@13-@32) */
  w13 -= w32;
  /* #99: @32 = sq(@13) */
  w32 = casadi_sq( w13 );
  /* #100: @33 = 0.16 */
  w33 = 1.6000000000000003e-01;
  /* #101: @32 = (@32/@33) */
  w32 /= w33;
  /* #102: @31 = (@31+@32) */
  w31 += w32;
  /* #103: @31 = (@23*@31) */
  w31  = (w23*w31);
  /* #104: @30 = (@30-@31) */
  w30 -= w31;
  /* #105: @31 = sq(@30) */
  w31 = casadi_sq( w30 );
  /* #106: @29 = (@29+@31) */
  w29 += w31;
  /* #107: @6 = (@6/@29) */
  w6 /= w29;
  /* #108: @31 = (@23*@6) */
  w31  = (w23*w6);
  /* #109: @32 = (@27*@31) */
  w32  = (w27*w31);
  /* #110: @33 = (2.*@18) */
  w33 = (2.* w18 );
  /* #111: @32 = (@32*@33) */
  w32 *= w33;
  /* #112: @33 = (2.*@9) */
  w33 = (2.* w9 );
  /* #113: @6 = (@6/@29) */
  w6 /= w29;
  /* #114: @30 = (2.*@30) */
  w30 = (2.* w30 );
  /* #115: @29 = 6.25 */
  w29 = 6.2499999999999991e+00;
  /* #116: @9 = (2.*@9) */
  w9 = (2.* w9 );
  /* #117: @9 = (@9*@18) */
  w9 *= w18;
  /* #118: @29 = (@29*@9) */
  w29 *= w9;
  /* #119: @29 = (@23*@29) */
  w29  = (w23*w29);
  /* #120: @29 = (@30*@29) */
  w29  = (w30*w29);
  /* #121: @29 = (@6*@29) */
  w29  = (w6*w29);
  /* #122: @29 = (@23*@29) */
  w29  = (w23*w29);
  /* #123: @9 = (@27*@29) */
  w9  = (w27*w29);
  /* #124: @9 = (@33*@9) */
  w9  = (w33*w9);
  /* #125: @32 = (@32+@9) */
  w32 += w9;
  /* #126: @16 = (@16-@32) */
  w16 -= w32;
  /* #127: @32 = (2.*@14) */
  w32 = (2.* w14 );
  /* #128: @9 = 6.25 */
  w9 = 6.2499999999999991e+00;
  /* #129: @7 = (@9*@7) */
  w7  = (w9*w7);
  /* #130: @7 = (@32*@7) */
  w7  = (w32*w7);
  /* #131: @7 = (-@7) */
  w7 = (- w7 );
  /* #132: @18 = (2.*@25) */
  w18 = (2.* w25 );
  /* #133: @34 = 6.25 */
  w34 = 6.2499999999999991e+00;
  /* #134: @21 = (@34*@21) */
  w21  = (w34*w21);
  /* #135: @21 = (@18*@21) */
  w21  = (w18*w21);
  /* #136: @7 = (@7-@21) */
  w7 -= w21;
  /* #137: @21 = (2.*@13) */
  w21 = (2.* w13 );
  /* #138: @35 = 6.25 */
  w35 = 6.2499999999999991e+00;
  /* #139: @29 = (@35*@29) */
  w29  = (w35*w29);
  /* #140: @29 = (@21*@29) */
  w29  = (w21*w29);
  /* #141: @7 = (@7-@29) */
  w7 -= w29;
  /* #142: @36 = 00 */
  /* #143: @37 = 00 */
  /* #144: @38 = vertcat(@1, @2, @16, @7, @36, @37) */
  rr=w38;
  *rr++ = w16;
  *rr++ = w7;
  /* #145: @39 = @38[:2] */
  for (rr=w39, ss=w38+0; ss!=w38+2; ss+=1) *rr++ = *ss;
  /* #146: (@0[:4:2] = @39) */
  for (rr=w0+0, ss=w39; rr!=w0+4; rr+=2) *rr = *ss++;
  /* #147: @1 = 00 */
  /* #148: @2 = 00 */
  /* #149: @16 = 6.25 */
  w16 = 6.2499999999999991e+00;
  /* #150: @14 = (2.*@14) */
  w14 = (2.* w14 );
  /* #151: @7 = ones(6x1,1nz) */
  w7 = 1.;
  /* #152: {NULL, NULL, NULL, @29, NULL, NULL} = vertsplit(@7) */
  w29 = w7;
  /* #153: @14 = (@14*@29) */
  w14 *= w29;
  /* #154: @16 = (@16*@14) */
  w16 *= w14;
  /* #155: @16 = (@4*@16) */
  w16  = (w4*w16);
  /* #156: @8 = (@8*@16) */
  w8 *= w16;
  /* #157: @12 = (@12*@8) */
  w12 *= w8;
  /* #158: @4 = (@4*@12) */
  w4 *= w12;
  /* #159: @3 = (@3*@4) */
  w3 *= w4;
  /* #160: @19 = (@19*@3) */
  w19 *= w3;
  /* #161: @19 = (-@19) */
  w19 = (- w19 );
  /* #162: @3 = 6.25 */
  w3 = 6.2499999999999991e+00;
  /* #163: @25 = (2.*@25) */
  w25 = (2.* w25 );
  /* #164: @25 = (@25*@29) */
  w25 *= w29;
  /* #165: @3 = (@3*@25) */
  w3 *= w25;
  /* #166: @3 = (@20*@3) */
  w3  = (w20*w3);
  /* #167: @22 = (@22*@3) */
  w22 *= w3;
  /* #168: @24 = (@24*@22) */
  w24 *= w22;
  /* #169: @20 = (@20*@24) */
  w20 *= w24;
  /* #170: @11 = (@11*@20) */
  w11 *= w20;
  /* #171: @28 = (@28*@11) */
  w28 *= w11;
  /* #172: @19 = (@19-@28) */
  w19 -= w28;
  /* #173: @28 = 6.25 */
  w28 = 6.2499999999999991e+00;
  /* #174: @13 = (2.*@13) */
  w13 = (2.* w13 );
  /* #175: @13 = (@13*@29) */
  w13 *= w29;
  /* #176: @28 = (@28*@13) */
  w28 *= w13;
  /* #177: @28 = (@23*@28) */
  w28  = (w23*w28);
  /* #178: @30 = (@30*@28) */
  w30 *= w28;
  /* #179: @6 = (@6*@30) */
  w6 *= w30;
  /* #180: @23 = (@23*@6) */
  w23 *= w6;
  /* #181: @27 = (@27*@23) */
  w27 *= w23;
  /* #182: @33 = (@33*@27) */
  w33 *= w27;
  /* #183: @19 = (@19-@33) */
  w19 -= w33;
  /* #184: @15 = (@9*@15) */
  w15  = (w9*w15);
  /* #185: @33 = (2.*@29) */
  w33 = (2.* w29 );
  /* #186: @15 = (@15*@33) */
  w15 *= w33;
  /* #187: @9 = (@9*@4) */
  w9 *= w4;
  /* #188: @32 = (@32*@9) */
  w32 *= w9;
  /* #189: @15 = (@15+@32) */
  w15 += w32;
  /* #190: @15 = (-@15) */
  w15 = (- w15 );
  /* #191: @26 = (@34*@26) */
  w26  = (w34*w26);
  /* #192: @32 = (2.*@29) */
  w32 = (2.* w29 );
  /* #193: @26 = (@26*@32) */
  w26 *= w32;
  /* #194: @34 = (@34*@20) */
  w34 *= w20;
  /* #195: @18 = (@18*@34) */
  w18 *= w34;
  /* #196: @26 = (@26+@18) */
  w26 += w18;
  /* #197: @15 = (@15-@26) */
  w15 -= w26;
  /* #198: @31 = (@35*@31) */
  w31  = (w35*w31);
  /* #199: @29 = (2.*@29) */
  w29 = (2.* w29 );
  /* #200: @31 = (@31*@29) */
  w31 *= w29;
  /* #201: @35 = (@35*@23) */
  w35 *= w23;
  /* #202: @21 = (@21*@35) */
  w21 *= w35;
  /* #203: @31 = (@31+@21) */
  w31 += w21;
  /* #204: @15 = (@15-@31) */
  w15 -= w31;
  /* #205: @36 = 00 */
  /* #206: @37 = 00 */
  /* #207: @39 = vertcat(@1, @2, @19, @15, @36, @37) */
  rr=w39;
  *rr++ = w19;
  *rr++ = w15;
  /* #208: @38 = @39[:2] */
  for (rr=w38, ss=w39+0; ss!=w39+2; ss+=1) *rr++ = *ss;
  /* #209: (@0[1:5:2] = @38) */
  for (rr=w0+1, ss=w38; rr!=w0+5; rr+=2) *rr = *ss++;
  /* #210: @40 = @0' */
  casadi_trans(w0,casadi_s0, w40, casadi_s0, iw);
  /* #211: output[0][0] = @40 */
  casadi_copy(w40, 4, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int mobile_robot_cost_y_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int mobile_robot_cost_y_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int mobile_robot_cost_y_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mobile_robot_cost_y_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int mobile_robot_cost_y_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mobile_robot_cost_y_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void mobile_robot_cost_y_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void mobile_robot_cost_y_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int mobile_robot_cost_y_hess_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int mobile_robot_cost_y_hess_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real mobile_robot_cost_y_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mobile_robot_cost_y_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mobile_robot_cost_y_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mobile_robot_cost_y_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s2;
    case 2: return casadi_s3;
    case 3: return casadi_s4;
    case 4: return casadi_s3;
    case 5: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mobile_robot_cost_y_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int mobile_robot_cost_y_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 12;
  if (sz_res) *sz_res = 8;
  if (sz_iw) *sz_iw = 7;
  if (sz_w) *sz_w = 60;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
