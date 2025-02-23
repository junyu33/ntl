#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#ifdef __x86_64__
#include <immintrin.h>
#define __int128 __m128i

inline void gfmul8(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 56) | (tmp3 >> 8); 
    tmp3 = tmp3 << 56 >> 56;

    tmp7 = tmp6 >> 7;
    tmp8 = tmp6 >> 5;
    tmp9 = tmp6 >> 4;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 56 >> 56;
}

inline void gfmul9(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 55) | (tmp3 >> 9); 
    tmp3 = tmp3 << 55 >> 55;

    tmp7 = tmp6 >> 8;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 55 >> 55;
}

inline void gfmul10(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 54) | (tmp3 >> 10); 
    tmp3 = tmp3 << 54 >> 54;

    tmp7 = tmp6 >> 7;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 54 >> 54;
}

inline void gfmul11(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 53) | (tmp3 >> 11); 
    tmp3 = tmp3 << 53 >> 53;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 53 >> 53;
}

inline void gfmul12(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 52) | (tmp3 >> 12); 
    tmp3 = tmp3 << 52 >> 52;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 52 >> 52;
}

inline void gfmul13(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 51) | (tmp3 >> 13); 
    tmp3 = tmp3 << 51 >> 51;

    tmp7 = tmp6 >> 12;
    tmp8 = tmp6 >> 10;
    tmp9 = tmp6 >> 9;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 51 >> 51;
}

inline void gfmul14(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 50) | (tmp3 >> 14); 
    tmp3 = tmp3 << 50 >> 50;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 50 >> 50;
}

inline void gfmul15(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 49) | (tmp3 >> 15); 
    tmp3 = tmp3 << 49 >> 49;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 49 >> 49;
}

inline void gfmul16(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 48) | (tmp3 >> 16); 
    tmp3 = tmp3 << 48 >> 48;

    tmp7 = tmp6 >> 15;
    tmp8 = tmp6 >> 13;
    tmp9 = tmp6 >> 11;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 48 >> 48;
}

inline void gfmul17(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 47) | (tmp3 >> 17); 
    tmp3 = tmp3 << 47 >> 47;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 47 >> 47;
}

inline void gfmul18(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 46) | (tmp3 >> 18); 
    tmp3 = tmp3 << 46 >> 46;

    tmp7 = tmp6 >> 15;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 46 >> 46;
}

inline void gfmul19(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 45) | (tmp3 >> 19); 
    tmp3 = tmp3 << 45 >> 45;

    tmp7 = tmp6 >> 18;
    tmp8 = tmp6 >> 17;
    tmp9 = tmp6 >> 14;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 45 >> 45;
}

inline void gfmul20(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 44) | (tmp3 >> 20); 
    tmp3 = tmp3 << 44 >> 44;

    tmp7 = tmp6 >> 17;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 44 >> 44;
}

inline void gfmul21(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 43) | (tmp3 >> 21); 
    tmp3 = tmp3 << 43 >> 43;

    tmp7 = tmp6 >> 19;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 43 >> 43;
}

inline void gfmul22(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 42) | (tmp3 >> 22); 
    tmp3 = tmp3 << 42 >> 42;

    tmp7 = tmp6 >> 21;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 42 >> 42;
}

inline void gfmul23(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 41) | (tmp3 >> 23); 
    tmp3 = tmp3 << 41 >> 41;

    tmp7 = tmp6 >> 18;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 41 >> 41;
}

inline void gfmul24(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 40) | (tmp3 >> 24); 
    tmp3 = tmp3 << 40 >> 40;

    tmp7 = tmp6 >> 23;
    tmp8 = tmp6 >> 21;
    tmp9 = tmp6 >> 20;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 40 >> 40;
}

inline void gfmul25(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 39) | (tmp3 >> 25); 
    tmp3 = tmp3 << 39 >> 39;

    tmp7 = tmp6 >> 22;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 39 >> 39;
}

inline void gfmul26(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 38) | (tmp3 >> 26); 
    tmp3 = tmp3 << 38 >> 38;

    tmp7 = tmp6 >> 25;
    tmp8 = tmp6 >> 23;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 38 >> 38;
}

inline void gfmul27(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 37) | (tmp3 >> 27); 
    tmp3 = tmp3 << 37 >> 37;

    tmp7 = tmp6 >> 26;
    tmp8 = tmp6 >> 25;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 37 >> 37;
}

inline void gfmul28(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 36) | (tmp3 >> 28); 
    tmp3 = tmp3 << 36 >> 36;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 36 >> 36;
}

inline void gfmul29(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 35) | (tmp3 >> 29); 
    tmp3 = tmp3 << 35 >> 35;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 35 >> 35;
}

inline void gfmul30(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 34) | (tmp3 >> 30); 
    tmp3 = tmp3 << 34 >> 34;

    tmp7 = tmp6 >> 29;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 34 >> 34;
}

inline void gfmul31(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 33) | (tmp3 >> 31); 
    tmp3 = tmp3 << 33 >> 33;

    tmp7 = tmp6 >> 28;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 33 >> 33;
}

inline void gfmul32(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 32) | (tmp3 >> 32); 
    tmp3 = tmp3 << 32 >> 32;

    tmp7 = tmp6 >> 30;
    tmp8 = tmp6 >> 29;
    tmp9 = tmp6 >> 25;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 32 >> 32;
}

inline void gfmul33(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 31) | (tmp3 >> 33); 
    tmp3 = tmp3 << 31 >> 31;

    tmp7 = tmp6 >> 23;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 10;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 31 >> 31;
}

inline void gfmul34(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 30) | (tmp3 >> 34); 
    tmp3 = tmp3 << 30 >> 30;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 30 >> 30;
}

inline void gfmul35(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 29) | (tmp3 >> 35); 
    tmp3 = tmp3 << 29 >> 29;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 29 >> 29;
}

inline void gfmul36(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 28) | (tmp3 >> 36); 
    tmp3 = tmp3 << 28 >> 28;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 28 >> 28;
}

inline void gfmul37(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 27) | (tmp3 >> 37); 
    tmp3 = tmp3 << 27 >> 27;

    tmp7 = tmp6 >> 36;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 31;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 27 >> 27;
}

inline void gfmul38(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 26) | (tmp3 >> 38); 
    tmp3 = tmp3 << 26 >> 26;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 32;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 5;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 26 >> 26;
}

inline void gfmul39(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 25) | (tmp3 >> 39); 
    tmp3 = tmp3 << 25 >> 25;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 25 >> 25;
}

inline void gfmul40(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 24) | (tmp3 >> 40); 
    tmp3 = tmp3 << 24 >> 24;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 36;
    tmp9 = tmp6 >> 35;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 24 >> 24;
}

inline void gfmul41(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 23) | (tmp3 >> 41); 
    tmp3 = tmp3 << 23 >> 23;

    tmp7 = tmp6 >> 38;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 23 >> 23;
}

inline void gfmul42(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 22) | (tmp3 >> 42); 
    tmp3 = tmp3 << 22 >> 22;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 22 >> 22;
}

inline void gfmul43(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 21) | (tmp3 >> 43); 
    tmp3 = tmp3 << 21 >> 21;

    tmp7 = tmp6 >> 40;
    tmp8 = tmp6 >> 39;
    tmp9 = tmp6 >> 37;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 21 >> 21;
}

inline void gfmul44(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 20) | (tmp3 >> 44); 
    tmp3 = tmp3 << 20 >> 20;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 20 >> 20;
}

inline void gfmul45(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 19) | (tmp3 >> 45); 
    tmp3 = tmp3 << 19 >> 19;

    tmp7 = tmp6 >> 44;
    tmp8 = tmp6 >> 42;
    tmp9 = tmp6 >> 41;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 19 >> 19;
}

inline void gfmul46(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 18) | (tmp3 >> 46); 
    tmp3 = tmp3 << 18 >> 18;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 18 >> 18;
}

inline void gfmul47(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 17) | (tmp3 >> 47); 
    tmp3 = tmp3 << 17 >> 17;

    tmp7 = tmp6 >> 42;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 17 >> 17;
}

inline void gfmul48(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 16) | (tmp3 >> 48); 
    tmp3 = tmp3 << 16 >> 16;

    tmp7 = tmp6 >> 46;
    tmp8 = tmp6 >> 45;
    tmp9 = tmp6 >> 43;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 16 >> 16;
}

inline void gfmul49(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 15) | (tmp3 >> 49); 
    tmp3 = tmp3 << 15 >> 15;

    tmp7 = tmp6 >> 40;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 15 >> 15;
}

inline void gfmul50(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 14) | (tmp3 >> 50); 
    tmp3 = tmp3 << 14 >> 14;

    tmp7 = tmp6 >> 48;
    tmp8 = tmp6 >> 47;
    tmp9 = tmp6 >> 46;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 14 >> 14;
}

inline void gfmul51(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 13) | (tmp3 >> 51); 
    tmp3 = tmp3 << 13 >> 13;

    tmp7 = tmp6 >> 50;
    tmp8 = tmp6 >> 48;
    tmp9 = tmp6 >> 45;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 13 >> 13;
}

inline void gfmul52(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 12) | (tmp3 >> 52); 
    tmp3 = tmp3 << 12 >> 12;

    tmp7 = tmp6 >> 49;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 12 >> 12;
}

inline void gfmul53(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 11) | (tmp3 >> 53); 
    tmp3 = tmp3 << 11 >> 11;

    tmp7 = tmp6 >> 52;
    tmp8 = tmp6 >> 51;
    tmp9 = tmp6 >> 47;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 11 >> 11;
}

inline void gfmul54(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 10) | (tmp3 >> 54); 
    tmp3 = tmp3 << 10 >> 10;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 10 >> 10;
}

inline void gfmul55(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 9) | (tmp3 >> 55); 
    tmp3 = tmp3 << 9 >> 9;

    tmp7 = tmp6 >> 48;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 9 >> 9;
}

inline void gfmul56(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 8) | (tmp3 >> 56); 
    tmp3 = tmp3 << 8 >> 8;

    tmp7 = tmp6 >> 54;
    tmp8 = tmp6 >> 52;
    tmp9 = tmp6 >> 49;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 8 >> 8;
}

inline void gfmul57(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 7) | (tmp3 >> 57); 
    tmp3 = tmp3 << 7 >> 7;

    tmp7 = tmp6 >> 53;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 7 >> 7;
}

inline void gfmul58(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 6) | (tmp3 >> 58); 
    tmp3 = tmp3 << 6 >> 6;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 19;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 6 >> 6;
}

inline void gfmul59(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 5) | (tmp3 >> 59); 
    tmp3 = tmp3 << 5 >> 5;

    tmp7 = tmp6 >> 57;
    tmp8 = tmp6 >> 55;
    tmp9 = tmp6 >> 52;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 5 >> 5;
}

inline void gfmul60(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 4) | (tmp3 >> 60); 
    tmp3 = tmp3 << 4 >> 4;

    tmp7 = tmp6 >> 59;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 4 >> 4;
}

inline void gfmul61(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 3) | (tmp3 >> 61); 
    tmp3 = tmp3 << 3 >> 3;

    tmp7 = tmp6 >> 60;
    tmp8 = tmp6 >> 59;
    tmp9 = tmp6 >> 56;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 3 >> 3;
}

inline void gfmul62(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 2) | (tmp3 >> 62); 
    tmp3 = tmp3 << 2 >> 2;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 29;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 2 >> 2;
}

inline void gfmul63(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp6 = (tmp6 << 1) | (tmp3 >> 63); 
    tmp3 = tmp3 << 1 >> 1;

    tmp7 = tmp6 >> 62;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 1 >> 1;
}

inline void gfmul64(__int128 a, __int128 b, uint64_t *res) {
    __int128 mul;
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = _mm_clmulepi64_si128(a, b, 0x00);
    tmp3 = _mm_extract_epi64(mul, 0);
    tmp6 = _mm_extract_epi64(mul, 1);

    tmp7 = tmp6 >> 63;
    tmp8 = tmp6 >> 61;
    tmp9 = tmp6 >> 60;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = tmp3 ^ tmp6;
}

inline void gfmul65(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 63) | (tmp3 >> 65); 
    tmp3 = tmp3 << 63 >> 63;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 47;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 63 >> 63;
}

inline void gfmul66(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 62) | (tmp3 >> 66); 
    tmp3 = tmp3 << 62 >> 62;

    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 62 >> 62;
}

inline void gfmul67(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 61) | (tmp3 >> 67); 
    tmp3 = tmp3 << 61 >> 61;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 65;
    tmp12 = tmp6 >> 66;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 61 >> 61;
}

inline void gfmul68(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 60) | (tmp3 >> 68); 
    tmp3 = tmp3 << 60 >> 60;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 59;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 60 >> 60;
}

inline void gfmul69(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 59) | (tmp3 >> 69); 
    tmp3 = tmp3 << 59 >> 59;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp11 = tmp6 >> 64;
    tmp12 = tmp6 >> 67;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 59 >> 59;
}

inline void gfmul70(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 58) | (tmp3 >> 70); 
    tmp3 = tmp3 << 58 >> 58;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp11 = tmp6 >> 67;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 58 >> 58;
}

inline void gfmul71(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 57) | (tmp3 >> 71); 
    tmp3 = tmp3 << 57 >> 57;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 57 >> 57;
}

inline void gfmul72(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 56) | (tmp3 >> 72); 
    tmp3 = tmp3 << 56 >> 56;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 63;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 56 >> 56;
}

inline void gfmul73(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 55) | (tmp3 >> 73); 
    tmp3 = tmp3 << 55 >> 55;

    tmp3 ^= tmp6 << 25;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 48;
    tmp10 = tmp10 ^ (tmp10 << 25);

    *res = tmp3 ^ tmp10;

    *res = *res << 55 >> 55;
}

inline void gfmul74(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 54) | (tmp3 >> 74); 
    tmp3 = tmp3 << 54 >> 54;

    tmp3 ^= tmp6 << 35;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 39;
    tmp10 = tmp10 ^ (tmp10 << 35);

    *res = tmp3 ^ tmp10;

    *res = *res << 54 >> 54;
}

inline void gfmul75(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 53) | (tmp3 >> 75); 
    tmp3 = tmp3 << 53 >> 53;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 69;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 74;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 53 >> 53;
}

inline void gfmul76(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 52) | (tmp3 >> 76); 
    tmp3 = tmp3 << 52 >> 52;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 55;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 52 >> 52;
}

inline void gfmul77(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 51) | (tmp3 >> 77); 
    tmp3 = tmp3 << 51 >> 51;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 51 >> 51;
}

inline void gfmul78(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 50) | (tmp3 >> 78); 
    tmp3 = tmp3 << 50 >> 50;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 72;
    tmp11 = tmp6 >> 73;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 50 >> 50;
}

inline void gfmul79(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 49) | (tmp3 >> 79); 
    tmp3 = tmp3 << 49 >> 49;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 70;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 49 >> 49;
}

inline void gfmul80(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 48) | (tmp3 >> 80); 
    tmp3 = tmp3 << 48 >> 48;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 76;
    tmp12 = tmp6 >> 78;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 48 >> 48;
}

inline void gfmul81(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 47) | (tmp3 >> 81); 
    tmp3 = tmp3 << 47 >> 47;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 47 >> 47;
}

inline void gfmul82(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 46) | (tmp3 >> 82); 
    tmp3 = tmp3 << 46 >> 46;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 46 >> 46;
}

inline void gfmul83(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 45) | (tmp3 >> 83); 
    tmp3 = tmp3 << 45 >> 45;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 76;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 45 >> 45;
}

inline void gfmul84(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 44) | (tmp3 >> 84); 
    tmp3 = tmp3 << 44 >> 44;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 79;
    tmp10 = tmp10 ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 44 >> 44;
}

inline void gfmul85(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 43) | (tmp3 >> 85); 
    tmp3 = tmp3 << 43 >> 43;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp11 = tmp6 >> 83;
    tmp12 = tmp6 >> 84;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 43 >> 43;
}

inline void gfmul86(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 42) | (tmp3 >> 86); 
    tmp3 = tmp3 << 42 >> 42;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 42 >> 42;
}

inline void gfmul87(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 41) | (tmp3 >> 87); 
    tmp3 = tmp3 << 41 >> 41;

    tmp3 ^= tmp6 << 13;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp10 = tmp10 ^ (tmp10 << 13);

    *res = tmp3 ^ tmp10;

    *res = *res << 41 >> 41;
}

inline void gfmul88(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 40) | (tmp3 >> 88); 
    tmp3 = tmp3 << 40 >> 40;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 81;
    tmp11 = tmp6 >> 82;
    tmp12 = tmp6 >> 86;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 40 >> 40;
}

inline void gfmul89(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 39) | (tmp3 >> 89); 
    tmp3 = tmp3 << 39 >> 39;

    tmp3 ^= tmp6 << 38;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 51;
    tmp10 = tmp10 ^ (tmp10 << 38);

    *res = tmp3 ^ tmp10;

    *res = *res << 39 >> 39;
}

inline void gfmul90(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 38) | (tmp3 >> 90); 
    tmp3 = tmp3 << 38 >> 38;

    tmp3 ^= tmp6 << 27;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 27);

    *res = tmp3 ^ tmp10;

    *res = *res << 38 >> 38;
}

inline void gfmul91(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 37) | (tmp3 >> 91); 
    tmp3 = tmp3 << 37 >> 37;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 83;
    tmp11 = tmp6 >> 86;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 5) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 37 >> 37;
}

inline void gfmul92(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 36) | (tmp3 >> 92); 
    tmp3 = tmp3 << 36 >> 36;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 36 >> 36;
}

inline void gfmul93(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 35) | (tmp3 >> 93); 
    tmp3 = tmp3 << 35 >> 35;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 35 >> 35;
}

inline void gfmul94(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 34) | (tmp3 >> 94); 
    tmp3 = tmp3 << 34 >> 34;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 34 >> 34;
}

inline void gfmul95(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 33) | (tmp3 >> 95); 
    tmp3 = tmp3 << 33 >> 33;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 84;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 33 >> 33;
}

inline void gfmul96(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 32) | (tmp3 >> 96); 
    tmp3 = tmp3 << 32 >> 32;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 86;
    tmp11 = tmp6 >> 87;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 32 >> 32;
}

inline void gfmul97(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 31) | (tmp3 >> 97); 
    tmp3 = tmp3 << 31 >> 31;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 31 >> 31;
}

inline void gfmul98(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 30) | (tmp3 >> 98); 
    tmp3 = tmp3 << 30 >> 30;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 87;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 30 >> 30;
}

inline void gfmul99(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 29) | (tmp3 >> 99); 
    tmp3 = tmp3 << 29 >> 29;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 93;
    tmp11 = tmp6 >> 96;
    tmp12 = tmp6 >> 98;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 29 >> 29;
}

inline void gfmul100(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 28) | (tmp3 >> 100); 
    tmp3 = tmp3 << 28 >> 28;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 28 >> 28;
}

inline void gfmul101(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 27) | (tmp3 >> 101); 
    tmp3 = tmp3 << 27 >> 27;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp11 = tmp6 >> 95;
    tmp12 = tmp6 >> 100;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 27 >> 27;
}

inline void gfmul102(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 26) | (tmp3 >> 102); 
    tmp3 = tmp3 << 26 >> 26;

    tmp3 ^= tmp6 << 29;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 29);

    *res = tmp3 ^ tmp10;

    *res = *res << 26 >> 26;
}

inline void gfmul103(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 25) | (tmp3 >> 103); 
    tmp3 = tmp3 << 25 >> 25;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 25 >> 25;
}

inline void gfmul104(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 24) | (tmp3 >> 104); 
    tmp3 = tmp3 << 24 >> 24;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 100;
    tmp11 = tmp6 >> 101;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 24 >> 24;
}

inline void gfmul105(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 23) | (tmp3 >> 105); 
    tmp3 = tmp3 << 23 >> 23;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 23 >> 23;
}

inline void gfmul106(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 22) | (tmp3 >> 106); 
    tmp3 = tmp3 << 22 >> 22;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 22 >> 22;
}

inline void gfmul107(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 21) | (tmp3 >> 107); 
    tmp3 = tmp3 << 21 >> 21;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 98;
    tmp11 = tmp6 >> 100;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 7) ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 21 >> 21;
}

inline void gfmul108(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 20) | (tmp3 >> 108); 
    tmp3 = tmp3 << 20 >> 20;

    tmp3 ^= tmp6 << 17;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 17);

    *res = tmp3 ^ tmp10;

    *res = *res << 20 >> 20;
}

inline void gfmul109(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 19) | (tmp3 >> 109); 
    tmp3 = tmp3 << 19 >> 19;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp11 = tmp6 >> 105;
    tmp12 = tmp6 >> 107;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 19 >> 19;
}

inline void gfmul110(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 18) | (tmp3 >> 110); 
    tmp3 = tmp3 << 18 >> 18;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 18 >> 18;
}

inline void gfmul111(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 17) | (tmp3 >> 111); 
    tmp3 = tmp3 << 17 >> 17;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 10);

    *res = tmp3 ^ tmp10;

    *res = *res << 17 >> 17;
}

inline void gfmul112(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 16) | (tmp3 >> 112); 
    tmp3 = tmp3 << 16 >> 16;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 109;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 16 >> 16;
}

inline void gfmul113(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 15) | (tmp3 >> 113); 
    tmp3 = tmp3 << 15 >> 15;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 15 >> 15;
}

inline void gfmul114(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 14) | (tmp3 >> 114); 
    tmp3 = tmp3 << 14 >> 14;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 109;
    tmp11 = tmp6 >> 111;
    tmp12 = tmp6 >> 112;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 14 >> 14;
}

inline void gfmul115(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 13) | (tmp3 >> 115); 
    tmp3 = tmp3 << 13 >> 13;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 110;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 7) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 13 >> 13;
}

inline void gfmul116(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 12) | (tmp3 >> 116); 
    tmp3 = tmp3 << 12 >> 12;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 114;
    tmp12 = tmp6 >> 115;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 12 >> 12;
}

inline void gfmul117(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 11) | (tmp3 >> 117); 
    tmp3 = tmp3 << 11 >> 11;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 115;
    tmp12 = tmp6 >> 116;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 11 >> 11;
}

inline void gfmul118(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 10) | (tmp3 >> 118); 
    tmp3 = tmp3 << 10 >> 10;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 10 >> 10;
}

inline void gfmul119(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 9) | (tmp3 >> 119); 
    tmp3 = tmp3 << 9 >> 9;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 111;
    tmp10 = tmp10 ^ (tmp10 << 8);

    *res = tmp3 ^ tmp10;

    *res = *res << 9 >> 9;
}

inline void gfmul120(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 8) | (tmp3 >> 120); 
    tmp3 = tmp3 << 8 >> 8;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 117;
    tmp12 = tmp6 >> 119;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 8 >> 8;
}

inline void gfmul121(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 7) | (tmp3 >> 121); 
    tmp3 = tmp3 << 7 >> 7;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 103;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 7 >> 7;
}

inline void gfmul122(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 6) | (tmp3 >> 122); 
    tmp3 = tmp3 << 6 >> 6;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 120;
    tmp12 = tmp6 >> 121;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 6 >> 6;
}

inline void gfmul123(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 5) | (tmp3 >> 123); 
    tmp3 = tmp3 << 5 >> 5;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 121;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 5 >> 5;
}

inline void gfmul124(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 4) | (tmp3 >> 124); 
    tmp3 = tmp3 << 4 >> 4;

    tmp3 ^= tmp6 << 19;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 19);

    *res = tmp3 ^ tmp10;

    *res = *res << 4 >> 4;
}

inline void gfmul125(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 3) | (tmp3 >> 125); 
    tmp3 = tmp3 << 3 >> 3;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 118;
    tmp11 = tmp6 >> 119;
    tmp12 = tmp6 >> 120;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 3 >> 3;
}

inline void gfmul126(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 2) | (tmp3 >> 126); 
    tmp3 = tmp3 << 2 >> 2;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 2 >> 2;
}

inline void gfmul127(__int128 a, __int128 b, __uint128_t *res){
    __int128 tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp7 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp8 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp7 = _mm_xor_si128(tmp7, tmp5);
    tmp8 = _mm_xor_si128(tmp8, tmp4);

    tmp3 = (__uint64_t)_mm_extract_epi64(tmp7, 0) | ((__uint128_t)_mm_extract_epi64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)_mm_extract_epi64(tmp8, 0) | ((__uint128_t)_mm_extract_epi64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 1) | (tmp3 >> 127); 
    tmp3 = tmp3 << 1 >> 1;

    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 126;
    tmp10 = tmp10 ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 1 >> 1;
}

inline void gfmul128(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    __int128 XMMMASK = _mm_setr_epi32(0xffffffff, 0xffffffff, 0x0, 0x0);
    tmp3 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp6 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp5);
    tmp6 = _mm_xor_si128(tmp6, tmp4);

    tmp7 = _mm_srli_epi64(tmp6, 63);
    tmp8 = _mm_srli_epi64(tmp6, 62);
    tmp9 = _mm_srli_epi64(tmp6, 57);
    tmp7 = _mm_xor_si128(tmp7, tmp8);
    tmp7 = _mm_xor_si128(tmp7, tmp9);
    tmp8 = _mm_shuffle_epi32(tmp7, 0x4e);

    tmp7 = _mm_and_si128(XMMMASK, tmp8);
    tmp8 = _mm_andnot_si128(XMMMASK, tmp8);
    tmp3 = _mm_xor_si128(tmp3, tmp8);
    tmp6 = _mm_xor_si128(tmp6, tmp7);
    tmp10 = _mm_slli_epi64(tmp6, 1);
    tmp11 = _mm_slli_epi64(tmp6, 2);
    tmp12 = _mm_slli_epi64(tmp6, 7);
    tmp3 = _mm_xor_si128(tmp3, tmp10);
    tmp3 = _mm_xor_si128(tmp3, tmp11);
    tmp3 = _mm_xor_si128(tmp3, tmp12);

    *res = _mm_xor_si128(tmp3, tmp6);

}

#elif __aarch64__
#include <arm_neon.h>
#define __int128 uint64x2_t

inline void gfmul8(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 56) | (tmp3 >> 8); 
    tmp3 = tmp3 << 56 >> 56;

    tmp7 = tmp6 >> 7;
    tmp8 = tmp6 >> 5;
    tmp9 = tmp6 >> 4;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 56 >> 56;
}

inline void gfmul9(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 55) | (tmp3 >> 9); 
    tmp3 = tmp3 << 55 >> 55;

    tmp7 = tmp6 >> 8;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 55 >> 55;
}

inline void gfmul10(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 54) | (tmp3 >> 10); 
    tmp3 = tmp3 << 54 >> 54;

    tmp7 = tmp6 >> 7;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 54 >> 54;
}

inline void gfmul11(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 53) | (tmp3 >> 11); 
    tmp3 = tmp3 << 53 >> 53;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 53 >> 53;
}

inline void gfmul12(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 52) | (tmp3 >> 12); 
    tmp3 = tmp3 << 52 >> 52;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 52 >> 52;
}

inline void gfmul13(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 51) | (tmp3 >> 13); 
    tmp3 = tmp3 << 51 >> 51;

    tmp7 = tmp6 >> 12;
    tmp8 = tmp6 >> 10;
    tmp9 = tmp6 >> 9;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 51 >> 51;
}

inline void gfmul14(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 50) | (tmp3 >> 14); 
    tmp3 = tmp3 << 50 >> 50;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 50 >> 50;
}

inline void gfmul15(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 49) | (tmp3 >> 15); 
    tmp3 = tmp3 << 49 >> 49;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 49 >> 49;
}

inline void gfmul16(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 48) | (tmp3 >> 16); 
    tmp3 = tmp3 << 48 >> 48;

    tmp7 = tmp6 >> 15;
    tmp8 = tmp6 >> 13;
    tmp9 = tmp6 >> 11;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 48 >> 48;
}

inline void gfmul17(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 47) | (tmp3 >> 17); 
    tmp3 = tmp3 << 47 >> 47;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 47 >> 47;
}

inline void gfmul18(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 46) | (tmp3 >> 18); 
    tmp3 = tmp3 << 46 >> 46;

    tmp7 = tmp6 >> 15;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 46 >> 46;
}

inline void gfmul19(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 45) | (tmp3 >> 19); 
    tmp3 = tmp3 << 45 >> 45;

    tmp7 = tmp6 >> 18;
    tmp8 = tmp6 >> 17;
    tmp9 = tmp6 >> 14;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 45 >> 45;
}

inline void gfmul20(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 44) | (tmp3 >> 20); 
    tmp3 = tmp3 << 44 >> 44;

    tmp7 = tmp6 >> 17;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 44 >> 44;
}

inline void gfmul21(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 43) | (tmp3 >> 21); 
    tmp3 = tmp3 << 43 >> 43;

    tmp7 = tmp6 >> 19;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 43 >> 43;
}

inline void gfmul22(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 42) | (tmp3 >> 22); 
    tmp3 = tmp3 << 42 >> 42;

    tmp7 = tmp6 >> 21;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 42 >> 42;
}

inline void gfmul23(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 41) | (tmp3 >> 23); 
    tmp3 = tmp3 << 41 >> 41;

    tmp7 = tmp6 >> 18;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 41 >> 41;
}

inline void gfmul24(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 40) | (tmp3 >> 24); 
    tmp3 = tmp3 << 40 >> 40;

    tmp7 = tmp6 >> 23;
    tmp8 = tmp6 >> 21;
    tmp9 = tmp6 >> 20;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 40 >> 40;
}

inline void gfmul25(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 39) | (tmp3 >> 25); 
    tmp3 = tmp3 << 39 >> 39;

    tmp7 = tmp6 >> 22;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 39 >> 39;
}

inline void gfmul26(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 38) | (tmp3 >> 26); 
    tmp3 = tmp3 << 38 >> 38;

    tmp7 = tmp6 >> 25;
    tmp8 = tmp6 >> 23;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 38 >> 38;
}

inline void gfmul27(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 37) | (tmp3 >> 27); 
    tmp3 = tmp3 << 37 >> 37;

    tmp7 = tmp6 >> 26;
    tmp8 = tmp6 >> 25;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 37 >> 37;
}

inline void gfmul28(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 36) | (tmp3 >> 28); 
    tmp3 = tmp3 << 36 >> 36;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 36 >> 36;
}

inline void gfmul29(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 35) | (tmp3 >> 29); 
    tmp3 = tmp3 << 35 >> 35;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 35 >> 35;
}

inline void gfmul30(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 34) | (tmp3 >> 30); 
    tmp3 = tmp3 << 34 >> 34;

    tmp7 = tmp6 >> 29;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 34 >> 34;
}

inline void gfmul31(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 33) | (tmp3 >> 31); 
    tmp3 = tmp3 << 33 >> 33;

    tmp7 = tmp6 >> 28;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 33 >> 33;
}

inline void gfmul32(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 32) | (tmp3 >> 32); 
    tmp3 = tmp3 << 32 >> 32;

    tmp7 = tmp6 >> 30;
    tmp8 = tmp6 >> 29;
    tmp9 = tmp6 >> 25;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 32 >> 32;
}

inline void gfmul33(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 31) | (tmp3 >> 33); 
    tmp3 = tmp3 << 31 >> 31;

    tmp7 = tmp6 >> 23;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 10;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 31 >> 31;
}

inline void gfmul34(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 30) | (tmp3 >> 34); 
    tmp3 = tmp3 << 30 >> 30;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 30 >> 30;
}

inline void gfmul35(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 29) | (tmp3 >> 35); 
    tmp3 = tmp3 << 29 >> 29;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 29 >> 29;
}

inline void gfmul36(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 28) | (tmp3 >> 36); 
    tmp3 = tmp3 << 28 >> 28;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 28 >> 28;
}

inline void gfmul37(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 27) | (tmp3 >> 37); 
    tmp3 = tmp3 << 27 >> 27;

    tmp7 = tmp6 >> 36;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 31;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 27 >> 27;
}

inline void gfmul38(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 26) | (tmp3 >> 38); 
    tmp3 = tmp3 << 26 >> 26;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 32;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 5;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 26 >> 26;
}

inline void gfmul39(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 25) | (tmp3 >> 39); 
    tmp3 = tmp3 << 25 >> 25;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 25 >> 25;
}

inline void gfmul40(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 24) | (tmp3 >> 40); 
    tmp3 = tmp3 << 24 >> 24;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 36;
    tmp9 = tmp6 >> 35;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 24 >> 24;
}

inline void gfmul41(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 23) | (tmp3 >> 41); 
    tmp3 = tmp3 << 23 >> 23;

    tmp7 = tmp6 >> 38;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 23 >> 23;
}

inline void gfmul42(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 22) | (tmp3 >> 42); 
    tmp3 = tmp3 << 22 >> 22;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 22 >> 22;
}

inline void gfmul43(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 21) | (tmp3 >> 43); 
    tmp3 = tmp3 << 21 >> 21;

    tmp7 = tmp6 >> 40;
    tmp8 = tmp6 >> 39;
    tmp9 = tmp6 >> 37;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 21 >> 21;
}

inline void gfmul44(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 20) | (tmp3 >> 44); 
    tmp3 = tmp3 << 20 >> 20;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 20 >> 20;
}

inline void gfmul45(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 19) | (tmp3 >> 45); 
    tmp3 = tmp3 << 19 >> 19;

    tmp7 = tmp6 >> 44;
    tmp8 = tmp6 >> 42;
    tmp9 = tmp6 >> 41;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 19 >> 19;
}

inline void gfmul46(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 18) | (tmp3 >> 46); 
    tmp3 = tmp3 << 18 >> 18;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 18 >> 18;
}

inline void gfmul47(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 17) | (tmp3 >> 47); 
    tmp3 = tmp3 << 17 >> 17;

    tmp7 = tmp6 >> 42;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 17 >> 17;
}

inline void gfmul48(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 16) | (tmp3 >> 48); 
    tmp3 = tmp3 << 16 >> 16;

    tmp7 = tmp6 >> 46;
    tmp8 = tmp6 >> 45;
    tmp9 = tmp6 >> 43;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 16 >> 16;
}

inline void gfmul49(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 15) | (tmp3 >> 49); 
    tmp3 = tmp3 << 15 >> 15;

    tmp7 = tmp6 >> 40;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 15 >> 15;
}

inline void gfmul50(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 14) | (tmp3 >> 50); 
    tmp3 = tmp3 << 14 >> 14;

    tmp7 = tmp6 >> 48;
    tmp8 = tmp6 >> 47;
    tmp9 = tmp6 >> 46;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 14 >> 14;
}

inline void gfmul51(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 13) | (tmp3 >> 51); 
    tmp3 = tmp3 << 13 >> 13;

    tmp7 = tmp6 >> 50;
    tmp8 = tmp6 >> 48;
    tmp9 = tmp6 >> 45;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 13 >> 13;
}

inline void gfmul52(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 12) | (tmp3 >> 52); 
    tmp3 = tmp3 << 12 >> 12;

    tmp7 = tmp6 >> 49;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 12 >> 12;
}

inline void gfmul53(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 11) | (tmp3 >> 53); 
    tmp3 = tmp3 << 11 >> 11;

    tmp7 = tmp6 >> 52;
    tmp8 = tmp6 >> 51;
    tmp9 = tmp6 >> 47;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 11 >> 11;
}

inline void gfmul54(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 10) | (tmp3 >> 54); 
    tmp3 = tmp3 << 10 >> 10;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 10 >> 10;
}

inline void gfmul55(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 9) | (tmp3 >> 55); 
    tmp3 = tmp3 << 9 >> 9;

    tmp7 = tmp6 >> 48;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 9 >> 9;
}

inline void gfmul56(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 8) | (tmp3 >> 56); 
    tmp3 = tmp3 << 8 >> 8;

    tmp7 = tmp6 >> 54;
    tmp8 = tmp6 >> 52;
    tmp9 = tmp6 >> 49;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 8 >> 8;
}

inline void gfmul57(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 7) | (tmp3 >> 57); 
    tmp3 = tmp3 << 7 >> 7;

    tmp7 = tmp6 >> 53;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 7 >> 7;
}

inline void gfmul58(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 6) | (tmp3 >> 58); 
    tmp3 = tmp3 << 6 >> 6;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 19;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 6 >> 6;
}

inline void gfmul59(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 5) | (tmp3 >> 59); 
    tmp3 = tmp3 << 5 >> 5;

    tmp7 = tmp6 >> 57;
    tmp8 = tmp6 >> 55;
    tmp9 = tmp6 >> 52;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 5 >> 5;
}

inline void gfmul60(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 4) | (tmp3 >> 60); 
    tmp3 = tmp3 << 4 >> 4;

    tmp7 = tmp6 >> 59;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 4 >> 4;
}

inline void gfmul61(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 3) | (tmp3 >> 61); 
    tmp3 = tmp3 << 3 >> 3;

    tmp7 = tmp6 >> 60;
    tmp8 = tmp6 >> 59;
    tmp9 = tmp6 >> 56;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 3 >> 3;
}

inline void gfmul62(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 2) | (tmp3 >> 62); 
    tmp3 = tmp3 << 2 >> 2;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 29;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 2 >> 2;
}

inline void gfmul63(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp6 = (tmp6 << 1) | (tmp3 >> 63); 
    tmp3 = tmp3 << 1 >> 1;

    tmp7 = tmp6 >> 62;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 1 >> 1;
}

inline void gfmul64(__int128 a, __int128 b, uint64_t *res) {
    poly128_t mul; 
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    mul = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
    tmp3 = (uint64_t)(mul);           
    tmp6 = (uint64_t)(mul >> 64);  

    tmp7 = tmp6 >> 63;
    tmp8 = tmp6 >> 61;
    tmp9 = tmp6 >> 60;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = tmp3 ^ tmp6;
}

inline void gfmul65(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 63) | (tmp3 >> 65); 
    tmp3 = tmp3 << 63 >> 63;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 47;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 63 >> 63;
}

inline void gfmul66(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 62) | (tmp3 >> 66); 
    tmp3 = tmp3 << 62 >> 62;

    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 62 >> 62;
}

inline void gfmul67(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 61) | (tmp3 >> 67); 
    tmp3 = tmp3 << 61 >> 61;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 65;
    tmp12 = tmp6 >> 66;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 61 >> 61;
}

inline void gfmul68(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 60) | (tmp3 >> 68); 
    tmp3 = tmp3 << 60 >> 60;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 59;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 60 >> 60;
}

inline void gfmul69(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 59) | (tmp3 >> 69); 
    tmp3 = tmp3 << 59 >> 59;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp11 = tmp6 >> 64;
    tmp12 = tmp6 >> 67;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 59 >> 59;
}

inline void gfmul70(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 58) | (tmp3 >> 70); 
    tmp3 = tmp3 << 58 >> 58;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp11 = tmp6 >> 67;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 58 >> 58;
}

inline void gfmul71(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 57) | (tmp3 >> 71); 
    tmp3 = tmp3 << 57 >> 57;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 57 >> 57;
}

inline void gfmul72(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 56) | (tmp3 >> 72); 
    tmp3 = tmp3 << 56 >> 56;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 63;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 56 >> 56;
}

inline void gfmul73(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 55) | (tmp3 >> 73); 
    tmp3 = tmp3 << 55 >> 55;

    tmp3 ^= tmp6 << 25;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 48;
    tmp10 = tmp10 ^ (tmp10 << 25);

    *res = tmp3 ^ tmp10;

    *res = *res << 55 >> 55;
}

inline void gfmul74(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 54) | (tmp3 >> 74); 
    tmp3 = tmp3 << 54 >> 54;

    tmp3 ^= tmp6 << 35;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 39;
    tmp10 = tmp10 ^ (tmp10 << 35);

    *res = tmp3 ^ tmp10;

    *res = *res << 54 >> 54;
}

inline void gfmul75(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 53) | (tmp3 >> 75); 
    tmp3 = tmp3 << 53 >> 53;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 69;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 74;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 53 >> 53;
}

inline void gfmul76(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 52) | (tmp3 >> 76); 
    tmp3 = tmp3 << 52 >> 52;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 55;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 52 >> 52;
}

inline void gfmul77(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 51) | (tmp3 >> 77); 
    tmp3 = tmp3 << 51 >> 51;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 51 >> 51;
}

inline void gfmul78(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 50) | (tmp3 >> 78); 
    tmp3 = tmp3 << 50 >> 50;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 72;
    tmp11 = tmp6 >> 73;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 50 >> 50;
}

inline void gfmul79(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 49) | (tmp3 >> 79); 
    tmp3 = tmp3 << 49 >> 49;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 70;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 49 >> 49;
}

inline void gfmul80(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 48) | (tmp3 >> 80); 
    tmp3 = tmp3 << 48 >> 48;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 76;
    tmp12 = tmp6 >> 78;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 48 >> 48;
}

inline void gfmul81(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 47) | (tmp3 >> 81); 
    tmp3 = tmp3 << 47 >> 47;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 47 >> 47;
}

inline void gfmul82(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 46) | (tmp3 >> 82); 
    tmp3 = tmp3 << 46 >> 46;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 46 >> 46;
}

inline void gfmul83(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 45) | (tmp3 >> 83); 
    tmp3 = tmp3 << 45 >> 45;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 76;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 45 >> 45;
}

inline void gfmul84(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 44) | (tmp3 >> 84); 
    tmp3 = tmp3 << 44 >> 44;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 79;
    tmp10 = tmp10 ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 44 >> 44;
}

inline void gfmul85(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 43) | (tmp3 >> 85); 
    tmp3 = tmp3 << 43 >> 43;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp11 = tmp6 >> 83;
    tmp12 = tmp6 >> 84;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 43 >> 43;
}

inline void gfmul86(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 42) | (tmp3 >> 86); 
    tmp3 = tmp3 << 42 >> 42;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 42 >> 42;
}

inline void gfmul87(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 41) | (tmp3 >> 87); 
    tmp3 = tmp3 << 41 >> 41;

    tmp3 ^= tmp6 << 13;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp10 = tmp10 ^ (tmp10 << 13);

    *res = tmp3 ^ tmp10;

    *res = *res << 41 >> 41;
}

inline void gfmul88(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 40) | (tmp3 >> 88); 
    tmp3 = tmp3 << 40 >> 40;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 81;
    tmp11 = tmp6 >> 82;
    tmp12 = tmp6 >> 86;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 40 >> 40;
}

inline void gfmul89(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 39) | (tmp3 >> 89); 
    tmp3 = tmp3 << 39 >> 39;

    tmp3 ^= tmp6 << 38;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 51;
    tmp10 = tmp10 ^ (tmp10 << 38);

    *res = tmp3 ^ tmp10;

    *res = *res << 39 >> 39;
}

inline void gfmul90(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 38) | (tmp3 >> 90); 
    tmp3 = tmp3 << 38 >> 38;

    tmp3 ^= tmp6 << 27;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 27);

    *res = tmp3 ^ tmp10;

    *res = *res << 38 >> 38;
}

inline void gfmul91(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 37) | (tmp3 >> 91); 
    tmp3 = tmp3 << 37 >> 37;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 83;
    tmp11 = tmp6 >> 86;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 5) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 37 >> 37;
}

inline void gfmul92(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 36) | (tmp3 >> 92); 
    tmp3 = tmp3 << 36 >> 36;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 36 >> 36;
}

inline void gfmul93(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 35) | (tmp3 >> 93); 
    tmp3 = tmp3 << 35 >> 35;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 35 >> 35;
}

inline void gfmul94(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 34) | (tmp3 >> 94); 
    tmp3 = tmp3 << 34 >> 34;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 34 >> 34;
}

inline void gfmul95(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 33) | (tmp3 >> 95); 
    tmp3 = tmp3 << 33 >> 33;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 84;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 33 >> 33;
}

inline void gfmul96(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 32) | (tmp3 >> 96); 
    tmp3 = tmp3 << 32 >> 32;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 86;
    tmp11 = tmp6 >> 87;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 32 >> 32;
}

inline void gfmul97(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 31) | (tmp3 >> 97); 
    tmp3 = tmp3 << 31 >> 31;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 31 >> 31;
}

inline void gfmul98(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 30) | (tmp3 >> 98); 
    tmp3 = tmp3 << 30 >> 30;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 87;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 30 >> 30;
}

inline void gfmul99(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 29) | (tmp3 >> 99); 
    tmp3 = tmp3 << 29 >> 29;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 93;
    tmp11 = tmp6 >> 96;
    tmp12 = tmp6 >> 98;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 29 >> 29;
}

inline void gfmul100(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 28) | (tmp3 >> 100); 
    tmp3 = tmp3 << 28 >> 28;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 28 >> 28;
}

inline void gfmul101(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 27) | (tmp3 >> 101); 
    tmp3 = tmp3 << 27 >> 27;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp11 = tmp6 >> 95;
    tmp12 = tmp6 >> 100;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 27 >> 27;
}

inline void gfmul102(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 26) | (tmp3 >> 102); 
    tmp3 = tmp3 << 26 >> 26;

    tmp3 ^= tmp6 << 29;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 29);

    *res = tmp3 ^ tmp10;

    *res = *res << 26 >> 26;
}

inline void gfmul103(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 25) | (tmp3 >> 103); 
    tmp3 = tmp3 << 25 >> 25;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 25 >> 25;
}

inline void gfmul104(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 24) | (tmp3 >> 104); 
    tmp3 = tmp3 << 24 >> 24;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 100;
    tmp11 = tmp6 >> 101;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 24 >> 24;
}

inline void gfmul105(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 23) | (tmp3 >> 105); 
    tmp3 = tmp3 << 23 >> 23;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 23 >> 23;
}

inline void gfmul106(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 22) | (tmp3 >> 106); 
    tmp3 = tmp3 << 22 >> 22;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 22 >> 22;
}

inline void gfmul107(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 21) | (tmp3 >> 107); 
    tmp3 = tmp3 << 21 >> 21;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 98;
    tmp11 = tmp6 >> 100;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 7) ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 21 >> 21;
}

inline void gfmul108(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 20) | (tmp3 >> 108); 
    tmp3 = tmp3 << 20 >> 20;

    tmp3 ^= tmp6 << 17;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 17);

    *res = tmp3 ^ tmp10;

    *res = *res << 20 >> 20;
}

inline void gfmul109(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 19) | (tmp3 >> 109); 
    tmp3 = tmp3 << 19 >> 19;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp11 = tmp6 >> 105;
    tmp12 = tmp6 >> 107;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 19 >> 19;
}

inline void gfmul110(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 18) | (tmp3 >> 110); 
    tmp3 = tmp3 << 18 >> 18;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 18 >> 18;
}

inline void gfmul111(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 17) | (tmp3 >> 111); 
    tmp3 = tmp3 << 17 >> 17;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 10);

    *res = tmp3 ^ tmp10;

    *res = *res << 17 >> 17;
}

inline void gfmul112(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 16) | (tmp3 >> 112); 
    tmp3 = tmp3 << 16 >> 16;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 109;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 16 >> 16;
}

inline void gfmul113(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 15) | (tmp3 >> 113); 
    tmp3 = tmp3 << 15 >> 15;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 15 >> 15;
}

inline void gfmul114(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 14) | (tmp3 >> 114); 
    tmp3 = tmp3 << 14 >> 14;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 109;
    tmp11 = tmp6 >> 111;
    tmp12 = tmp6 >> 112;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 14 >> 14;
}

inline void gfmul115(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 13) | (tmp3 >> 115); 
    tmp3 = tmp3 << 13 >> 13;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 110;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 7) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 13 >> 13;
}

inline void gfmul116(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 12) | (tmp3 >> 116); 
    tmp3 = tmp3 << 12 >> 12;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 114;
    tmp12 = tmp6 >> 115;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 12 >> 12;
}

inline void gfmul117(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 11) | (tmp3 >> 117); 
    tmp3 = tmp3 << 11 >> 11;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 115;
    tmp12 = tmp6 >> 116;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 11 >> 11;
}

inline void gfmul118(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 10) | (tmp3 >> 118); 
    tmp3 = tmp3 << 10 >> 10;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 10 >> 10;
}

inline void gfmul119(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 9) | (tmp3 >> 119); 
    tmp3 = tmp3 << 9 >> 9;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 111;
    tmp10 = tmp10 ^ (tmp10 << 8);

    *res = tmp3 ^ tmp10;

    *res = *res << 9 >> 9;
}

inline void gfmul120(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 8) | (tmp3 >> 120); 
    tmp3 = tmp3 << 8 >> 8;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 117;
    tmp12 = tmp6 >> 119;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 8 >> 8;
}

inline void gfmul121(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 7) | (tmp3 >> 121); 
    tmp3 = tmp3 << 7 >> 7;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 103;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 7 >> 7;
}

inline void gfmul122(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 6) | (tmp3 >> 122); 
    tmp3 = tmp3 << 6 >> 6;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 120;
    tmp12 = tmp6 >> 121;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 6 >> 6;
}

inline void gfmul123(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 5) | (tmp3 >> 123); 
    tmp3 = tmp3 << 5 >> 5;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 121;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 5 >> 5;
}

inline void gfmul124(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 4) | (tmp3 >> 124); 
    tmp3 = tmp3 << 4 >> 4;

    tmp3 ^= tmp6 << 19;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 19);

    *res = tmp3 ^ tmp10;

    *res = *res << 4 >> 4;
}

inline void gfmul125(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 3) | (tmp3 >> 125); 
    tmp3 = tmp3 << 3 >> 3;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 118;
    tmp11 = tmp6 >> 119;
    tmp12 = tmp6 >> 120;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 3 >> 3;
}

inline void gfmul126(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 2) | (tmp3 >> 126); 
    tmp3 = tmp3 << 2 >> 2;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 2 >> 2;
}

inline void gfmul127(__int128 a, __int128 b, __uint128_t *res){
    uint64x2_t tmp7, tmp4, tmp5, tmp8;
    __uint128_t tmp3, tmp6, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp7 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp8 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp7 = veorq_u64(tmp7, tmp5);
    tmp8 = veorq_u64(tmp8, tmp4);

    tmp3 = (__uint64_t)vgetq_lane_u64(tmp7, 0) | ((__uint128_t)vgetq_lane_u64(tmp7, 1) << 64);
    tmp6 = (__uint64_t)vgetq_lane_u64(tmp8, 0) | ((__uint128_t)vgetq_lane_u64(tmp8, 1) << 64);

    tmp6 = (tmp6 << 1) | (tmp3 >> 127); 
    tmp3 = tmp3 << 1 >> 1;

    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 126;
    tmp10 = tmp10 ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 1 >> 1;
}

inline void gfmul128(__int128 a, __int128 b, __int128 *res){
    uint64x2_t tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64x2_t XMMMASK = {0xffffffffffffffff, 0x0};
    uint64x2_t INV_XMMMASK = {0x0, 0xffffffffffffffff};
    poly64_t a_lo = (poly64_t)vgetq_lane_u64(a, 0);
    poly64_t b_lo = (poly64_t)vgetq_lane_u64(b, 0);
    poly64_t a_hi = (poly64_t)vgetq_lane_u64(a, 1);
    poly64_t b_hi = (poly64_t)vgetq_lane_u64(b, 1);

    tmp3 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
    tmp4 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
    tmp5 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
    tmp6 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));

    tmp4 = veorq_u64(tmp4, tmp5);
    __int128 zero = vdupq_n_u64(0);
    tmp5 = vextq_u64(zero, tmp4, 1);
    tmp4 = vextq_u64(tmp4, zero, 1); 
    tmp3 = veorq_u64(tmp3, tmp5);
    tmp6 = veorq_u64(tmp6, tmp4);    

    tmp7 = vshrq_n_u64(tmp6, 63);
    tmp8 = vshrq_n_u64(tmp6, 62);
    tmp9 = vshrq_n_u64(tmp6, 57);
    tmp7 = veorq_u64(tmp7, tmp8);
    tmp7 = veorq_u64(tmp7, tmp9);
    tmp8 = vextq_u64(tmp7, tmp7, 1);

    tmp7 = vandq_u64(tmp8, XMMMASK);
    tmp8 = vandq_u64(tmp8, INV_XMMMASK);
    tmp3 = veorq_u64(tmp3, tmp8);
    tmp6 = veorq_u64(tmp6, tmp7);
    tmp10 = vshlq_n_u64(tmp6, 1);
    tmp11 = vshlq_n_u64(tmp6, 2);
    tmp12 = vshlq_n_u64(tmp6, 7);
    tmp3 = veorq_u64(tmp3, tmp10);
    tmp3 = veorq_u64(tmp3, tmp11);
    tmp3 = veorq_u64(tmp3, tmp12);

    *res = veorq_u64(tmp3, tmp6);

}
#elif __riscv
#include <riscv_vector.h>
#define __int128 __uint128_t


inline void gfmul8(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 56) | (tmp3 >> 8); 
    tmp3 = tmp3 << 56 >> 56;

    tmp7 = tmp6 >> 7;
    tmp8 = tmp6 >> 5;
    tmp9 = tmp6 >> 4;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 56 >> 56;
}

inline void gfmul9(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 55) | (tmp3 >> 9); 
    tmp3 = tmp3 << 55 >> 55;

    tmp7 = tmp6 >> 8;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 55 >> 55;
}

inline void gfmul10(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 54) | (tmp3 >> 10); 
    tmp3 = tmp3 << 54 >> 54;

    tmp7 = tmp6 >> 7;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 54 >> 54;
}

inline void gfmul11(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 53) | (tmp3 >> 11); 
    tmp3 = tmp3 << 53 >> 53;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 53 >> 53;
}

inline void gfmul12(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 52) | (tmp3 >> 12); 
    tmp3 = tmp3 << 52 >> 52;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 52 >> 52;
}

inline void gfmul13(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 51) | (tmp3 >> 13); 
    tmp3 = tmp3 << 51 >> 51;

    tmp7 = tmp6 >> 12;
    tmp8 = tmp6 >> 10;
    tmp9 = tmp6 >> 9;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 51 >> 51;
}

inline void gfmul14(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 50) | (tmp3 >> 14); 
    tmp3 = tmp3 << 50 >> 50;

    tmp7 = tmp6 >> 9;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 50 >> 50;
}

inline void gfmul15(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 49) | (tmp3 >> 15); 
    tmp3 = tmp3 << 49 >> 49;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 49 >> 49;
}

inline void gfmul16(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 48) | (tmp3 >> 16); 
    tmp3 = tmp3 << 48 >> 48;

    tmp7 = tmp6 >> 15;
    tmp8 = tmp6 >> 13;
    tmp9 = tmp6 >> 11;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 48 >> 48;
}

inline void gfmul17(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 47) | (tmp3 >> 17); 
    tmp3 = tmp3 << 47 >> 47;

    tmp7 = tmp6 >> 14;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 47 >> 47;
}

inline void gfmul18(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 46) | (tmp3 >> 18); 
    tmp3 = tmp3 << 46 >> 46;

    tmp7 = tmp6 >> 15;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 46 >> 46;
}

inline void gfmul19(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 45) | (tmp3 >> 19); 
    tmp3 = tmp3 << 45 >> 45;

    tmp7 = tmp6 >> 18;
    tmp8 = tmp6 >> 17;
    tmp9 = tmp6 >> 14;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 45 >> 45;
}

inline void gfmul20(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 44) | (tmp3 >> 20); 
    tmp3 = tmp3 << 44 >> 44;

    tmp7 = tmp6 >> 17;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 44 >> 44;
}

inline void gfmul21(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 43) | (tmp3 >> 21); 
    tmp3 = tmp3 << 43 >> 43;

    tmp7 = tmp6 >> 19;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 43 >> 43;
}

inline void gfmul22(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 42) | (tmp3 >> 22); 
    tmp3 = tmp3 << 42 >> 42;

    tmp7 = tmp6 >> 21;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 42 >> 42;
}

inline void gfmul23(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 41) | (tmp3 >> 23); 
    tmp3 = tmp3 << 41 >> 41;

    tmp7 = tmp6 >> 18;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 41 >> 41;
}

inline void gfmul24(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 40) | (tmp3 >> 24); 
    tmp3 = tmp3 << 40 >> 40;

    tmp7 = tmp6 >> 23;
    tmp8 = tmp6 >> 21;
    tmp9 = tmp6 >> 20;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 40 >> 40;
}

inline void gfmul25(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 39) | (tmp3 >> 25); 
    tmp3 = tmp3 << 39 >> 39;

    tmp7 = tmp6 >> 22;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 39 >> 39;
}

inline void gfmul26(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 38) | (tmp3 >> 26); 
    tmp3 = tmp3 << 38 >> 38;

    tmp7 = tmp6 >> 25;
    tmp8 = tmp6 >> 23;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 38 >> 38;
}

inline void gfmul27(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 37) | (tmp3 >> 27); 
    tmp3 = tmp3 << 37 >> 37;

    tmp7 = tmp6 >> 26;
    tmp8 = tmp6 >> 25;
    tmp9 = tmp6 >> 22;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 37 >> 37;
}

inline void gfmul28(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 36) | (tmp3 >> 28); 
    tmp3 = tmp3 << 36 >> 36;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 36 >> 36;
}

inline void gfmul29(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 35) | (tmp3 >> 29); 
    tmp3 = tmp3 << 35 >> 35;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 35 >> 35;
}

inline void gfmul30(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 34) | (tmp3 >> 30); 
    tmp3 = tmp3 << 34 >> 34;

    tmp7 = tmp6 >> 29;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 34 >> 34;
}

inline void gfmul31(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 33) | (tmp3 >> 31); 
    tmp3 = tmp3 << 33 >> 33;

    tmp7 = tmp6 >> 28;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 33 >> 33;
}

inline void gfmul32(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 32) | (tmp3 >> 32); 
    tmp3 = tmp3 << 32 >> 32;

    tmp7 = tmp6 >> 30;
    tmp8 = tmp6 >> 29;
    tmp9 = tmp6 >> 25;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 32 >> 32;
}

inline void gfmul33(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 31) | (tmp3 >> 33); 
    tmp3 = tmp3 << 31 >> 31;

    tmp7 = tmp6 >> 23;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 10;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 31 >> 31;
}

inline void gfmul34(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 30) | (tmp3 >> 34); 
    tmp3 = tmp3 << 30 >> 30;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 30 >> 30;
}

inline void gfmul35(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 29) | (tmp3 >> 35); 
    tmp3 = tmp3 << 29 >> 29;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 2;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 29 >> 29;
}

inline void gfmul36(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 28) | (tmp3 >> 36); 
    tmp3 = tmp3 << 28 >> 28;

    tmp7 = tmp6 >> 27;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 28 >> 28;
}

inline void gfmul37(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 27) | (tmp3 >> 37); 
    tmp3 = tmp3 << 27 >> 27;

    tmp7 = tmp6 >> 36;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 31;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 27 >> 27;
}

inline void gfmul38(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 26) | (tmp3 >> 38); 
    tmp3 = tmp3 << 26 >> 26;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 33;
    tmp9 = tmp6 >> 32;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 5;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 26 >> 26;
}

inline void gfmul39(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 25) | (tmp3 >> 39); 
    tmp3 = tmp3 << 25 >> 25;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 25 >> 25;
}

inline void gfmul40(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 24) | (tmp3 >> 40); 
    tmp3 = tmp3 << 24 >> 24;

    tmp7 = tmp6 >> 37;
    tmp8 = tmp6 >> 36;
    tmp9 = tmp6 >> 35;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 24 >> 24;
}

inline void gfmul41(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 23) | (tmp3 >> 41); 
    tmp3 = tmp3 << 23 >> 23;

    tmp7 = tmp6 >> 38;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 23 >> 23;
}

inline void gfmul42(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 22) | (tmp3 >> 42); 
    tmp3 = tmp3 << 22 >> 22;

    tmp7 = tmp6 >> 35;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 22 >> 22;
}

inline void gfmul43(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 21) | (tmp3 >> 43); 
    tmp3 = tmp3 << 21 >> 21;

    tmp7 = tmp6 >> 40;
    tmp8 = tmp6 >> 39;
    tmp9 = tmp6 >> 37;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 3;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 21 >> 21;
}

inline void gfmul44(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 20) | (tmp3 >> 44); 
    tmp3 = tmp3 << 20 >> 20;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 20 >> 20;
}

inline void gfmul45(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 19) | (tmp3 >> 45); 
    tmp3 = tmp3 << 19 >> 19;

    tmp7 = tmp6 >> 44;
    tmp8 = tmp6 >> 42;
    tmp9 = tmp6 >> 41;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 19 >> 19;
}

inline void gfmul46(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 18) | (tmp3 >> 46); 
    tmp3 = tmp3 << 18 >> 18;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 18 >> 18;
}

inline void gfmul47(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 17) | (tmp3 >> 47); 
    tmp3 = tmp3 << 17 >> 17;

    tmp7 = tmp6 >> 42;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 17 >> 17;
}

inline void gfmul48(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 16) | (tmp3 >> 48); 
    tmp3 = tmp3 << 16 >> 16;

    tmp7 = tmp6 >> 46;
    tmp8 = tmp6 >> 45;
    tmp9 = tmp6 >> 43;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 16 >> 16;
}

inline void gfmul49(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 15) | (tmp3 >> 49); 
    tmp3 = tmp3 << 15 >> 15;

    tmp7 = tmp6 >> 40;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 15 >> 15;
}

inline void gfmul50(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 14) | (tmp3 >> 50); 
    tmp3 = tmp3 << 14 >> 14;

    tmp7 = tmp6 >> 48;
    tmp8 = tmp6 >> 47;
    tmp9 = tmp6 >> 46;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 14 >> 14;
}

inline void gfmul51(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 13) | (tmp3 >> 51); 
    tmp3 = tmp3 << 13 >> 13;

    tmp7 = tmp6 >> 50;
    tmp8 = tmp6 >> 48;
    tmp9 = tmp6 >> 45;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 13 >> 13;
}

inline void gfmul52(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 12) | (tmp3 >> 52); 
    tmp3 = tmp3 << 12 >> 12;

    tmp7 = tmp6 >> 49;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 3;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 12 >> 12;
}

inline void gfmul53(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 11) | (tmp3 >> 53); 
    tmp3 = tmp3 << 11 >> 11;

    tmp7 = tmp6 >> 52;
    tmp8 = tmp6 >> 51;
    tmp9 = tmp6 >> 47;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 6;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 11 >> 11;
}

inline void gfmul54(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 10) | (tmp3 >> 54); 
    tmp3 = tmp3 << 10 >> 10;

    tmp7 = tmp6 >> 45;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 9;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 10 >> 10;
}

inline void gfmul55(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 9) | (tmp3 >> 55); 
    tmp3 = tmp3 << 9 >> 9;

    tmp7 = tmp6 >> 48;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 9 >> 9;
}

inline void gfmul56(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 8) | (tmp3 >> 56); 
    tmp3 = tmp3 << 8 >> 8;

    tmp7 = tmp6 >> 54;
    tmp8 = tmp6 >> 52;
    tmp9 = tmp6 >> 49;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 8 >> 8;
}

inline void gfmul57(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 7) | (tmp3 >> 57); 
    tmp3 = tmp3 << 7 >> 7;

    tmp7 = tmp6 >> 53;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 7 >> 7;
}

inline void gfmul58(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 6) | (tmp3 >> 58); 
    tmp3 = tmp3 << 6 >> 6;

    tmp7 = tmp6 >> 39;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 19;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 6 >> 6;
}

inline void gfmul59(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 5) | (tmp3 >> 59); 
    tmp3 = tmp3 << 5 >> 5;

    tmp7 = tmp6 >> 57;
    tmp8 = tmp6 >> 55;
    tmp9 = tmp6 >> 52;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 2;
    tmp11 = tmp6 << 4;
    tmp12 = tmp6 << 7;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 5 >> 5;
}

inline void gfmul60(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 4) | (tmp3 >> 60); 
    tmp3 = tmp3 << 4 >> 4;

    tmp7 = tmp6 >> 59;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 4 >> 4;
}

inline void gfmul61(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 3) | (tmp3 >> 61); 
    tmp3 = tmp3 << 3 >> 3;

    tmp7 = tmp6 >> 60;
    tmp8 = tmp6 >> 59;
    tmp9 = tmp6 >> 56;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 2;
    tmp12 = tmp6 << 5;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 3 >> 3;
}

inline void gfmul62(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 2) | (tmp3 >> 62); 
    tmp3 = tmp3 << 2 >> 2;

    tmp7 = tmp6 >> 33;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 29;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 2 >> 2;
}

inline void gfmul63(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp6 = (tmp6 << 1) | (tmp3 >> 63); 
    tmp3 = tmp3 << 1 >> 1;

    tmp7 = tmp6 >> 62;
    tmp6 = tmp6 ^ tmp7;
    tmp12 = tmp6 << 1;
    tmp3 = tmp3 ^ tmp12;

    *res = (tmp3 ^ tmp6) << 1 >> 1;
}

inline void gfmul64(__int128 a, __int128 b, uint64_t *res) {
    uint64_t tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

    __asm__ __volatile__ (
        "mv t0, %2\n"             
        "mv t1, %3\n"                    

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 

        : "=r" (tmp3), "=r" (tmp6)
        : "r" ((uint64_t)a), "r" ((uint64_t)b)
        : "t0", "t1"
    );

    tmp7 = tmp6 >> 63;
    tmp8 = tmp6 >> 61;
    tmp9 = tmp6 >> 60;
    tmp7 = tmp7 ^ tmp8;
    tmp7 = tmp7 ^ tmp9;

    tmp6 = tmp6 ^ tmp7;
    tmp10 = tmp6 << 1;
    tmp11 = tmp6 << 3;
    tmp12 = tmp6 << 4;
    tmp3 = tmp3 ^ tmp10;
    tmp3 = tmp3 ^ tmp11;
    tmp3 = tmp3 ^ tmp12;

    *res = tmp3 ^ tmp6;
}

inline void gfmul65(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 63) | (tmp3 >> 65); 
    tmp3 = tmp3 << 63 >> 63;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 47;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 63 >> 63;
}

inline void gfmul66(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 62) | (tmp3 >> 66); 
    tmp3 = tmp3 << 62 >> 62;

    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 62 >> 62;
}

inline void gfmul67(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 61) | (tmp3 >> 67); 
    tmp3 = tmp3 << 61 >> 61;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 65;
    tmp12 = tmp6 >> 66;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 61 >> 61;
}

inline void gfmul68(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 60) | (tmp3 >> 68); 
    tmp3 = tmp3 << 60 >> 60;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 59;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 60 >> 60;
}

inline void gfmul69(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 59) | (tmp3 >> 69); 
    tmp3 = tmp3 << 59 >> 59;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp11 = tmp6 >> 64;
    tmp12 = tmp6 >> 67;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 59 >> 59;
}

inline void gfmul70(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 58) | (tmp3 >> 70); 
    tmp3 = tmp3 << 58 >> 58;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp11 = tmp6 >> 67;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 58 >> 58;
}

inline void gfmul71(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 57) | (tmp3 >> 71); 
    tmp3 = tmp3 << 57 >> 57;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 57 >> 57;
}

inline void gfmul72(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 56) | (tmp3 >> 72); 
    tmp3 = tmp3 << 56 >> 56;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 62;
    tmp11 = tmp6 >> 63;
    tmp12 = tmp6 >> 69;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 56 >> 56;
}

inline void gfmul73(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 55) | (tmp3 >> 73); 
    tmp3 = tmp3 << 55 >> 55;

    tmp3 ^= tmp6 << 25;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 48;
    tmp10 = tmp10 ^ (tmp10 << 25);

    *res = tmp3 ^ tmp10;

    *res = *res << 55 >> 55;
}

inline void gfmul74(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 54) | (tmp3 >> 74); 
    tmp3 = tmp3 << 54 >> 54;

    tmp3 ^= tmp6 << 35;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 39;
    tmp10 = tmp10 ^ (tmp10 << 35);

    *res = tmp3 ^ tmp10;

    *res = *res << 54 >> 54;
}

inline void gfmul75(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 53) | (tmp3 >> 75); 
    tmp3 = tmp3 << 53 >> 53;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 69;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 74;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 53 >> 53;
}

inline void gfmul76(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 52) | (tmp3 >> 76); 
    tmp3 = tmp3 << 52 >> 52;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 55;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 52 >> 52;
}

inline void gfmul77(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 51) | (tmp3 >> 77); 
    tmp3 = tmp3 << 51 >> 51;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 72;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 51 >> 51;
}

inline void gfmul78(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 50) | (tmp3 >> 78); 
    tmp3 = tmp3 << 50 >> 50;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 72;
    tmp11 = tmp6 >> 73;
    tmp12 = tmp6 >> 75;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 5) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 50 >> 50;
}

inline void gfmul79(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 49) | (tmp3 >> 79); 
    tmp3 = tmp3 << 49 >> 49;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 70;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 49 >> 49;
}

inline void gfmul80(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 48) | (tmp3 >> 80); 
    tmp3 = tmp3 << 48 >> 48;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp11 = tmp6 >> 76;
    tmp12 = tmp6 >> 78;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 48 >> 48;
}

inline void gfmul81(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 47) | (tmp3 >> 81); 
    tmp3 = tmp3 << 47 >> 47;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 47 >> 47;
}

inline void gfmul82(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 46) | (tmp3 >> 82); 
    tmp3 = tmp3 << 46 >> 46;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 46 >> 46;
}

inline void gfmul83(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 45) | (tmp3 >> 83); 
    tmp3 = tmp3 << 45 >> 45;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 76;
    tmp11 = tmp6 >> 79;
    tmp12 = tmp6 >> 81;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 45 >> 45;
}

inline void gfmul84(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 44) | (tmp3 >> 84); 
    tmp3 = tmp3 << 44 >> 44;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 79;
    tmp10 = tmp10 ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 44 >> 44;
}

inline void gfmul85(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 43) | (tmp3 >> 85); 
    tmp3 = tmp3 << 43 >> 43;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp11 = tmp6 >> 83;
    tmp12 = tmp6 >> 84;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 43 >> 43;
}

inline void gfmul86(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 42) | (tmp3 >> 86); 
    tmp3 = tmp3 << 42 >> 42;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 65;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 42 >> 42;
}

inline void gfmul87(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 41) | (tmp3 >> 87); 
    tmp3 = tmp3 << 41 >> 41;

    tmp3 ^= tmp6 << 13;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 74;
    tmp10 = tmp10 ^ (tmp10 << 13);

    *res = tmp3 ^ tmp10;

    *res = *res << 41 >> 41;
}

inline void gfmul88(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 40) | (tmp3 >> 88); 
    tmp3 = tmp3 << 40 >> 40;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 81;
    tmp11 = tmp6 >> 82;
    tmp12 = tmp6 >> 86;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 40 >> 40;
}

inline void gfmul89(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 39) | (tmp3 >> 89); 
    tmp3 = tmp3 << 39 >> 39;

    tmp3 ^= tmp6 << 38;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 51;
    tmp10 = tmp10 ^ (tmp10 << 38);

    *res = tmp3 ^ tmp10;

    *res = *res << 39 >> 39;
}

inline void gfmul90(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 38) | (tmp3 >> 90); 
    tmp3 = tmp3 << 38 >> 38;

    tmp3 ^= tmp6 << 27;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 63;
    tmp10 = tmp10 ^ (tmp10 << 27);

    *res = tmp3 ^ tmp10;

    *res = *res << 38 >> 38;
}

inline void gfmul91(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 37) | (tmp3 >> 91); 
    tmp3 = tmp3 << 37 >> 37;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 83;
    tmp11 = tmp6 >> 86;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 5) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 37 >> 37;
}

inline void gfmul92(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 36) | (tmp3 >> 92); 
    tmp3 = tmp3 << 36 >> 36;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 71;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 36 >> 36;
}

inline void gfmul93(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 35) | (tmp3 >> 93); 
    tmp3 = tmp3 << 35 >> 35;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 35 >> 35;
}

inline void gfmul94(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 34) | (tmp3 >> 94); 
    tmp3 = tmp3 << 34 >> 34;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 34 >> 34;
}

inline void gfmul95(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 33) | (tmp3 >> 95); 
    tmp3 = tmp3 << 33 >> 33;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 84;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 33 >> 33;
}

inline void gfmul96(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 32) | (tmp3 >> 96); 
    tmp3 = tmp3 << 32 >> 32;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 86;
    tmp11 = tmp6 >> 87;
    tmp12 = tmp6 >> 90;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 10) ^ (tmp10 << 9) ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 32 >> 32;
}

inline void gfmul97(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 31) | (tmp3 >> 97); 
    tmp3 = tmp3 << 31 >> 31;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 6);

    *res = tmp3 ^ tmp10;

    *res = *res << 31 >> 31;
}

inline void gfmul98(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 30) | (tmp3 >> 98); 
    tmp3 = tmp3 << 30 >> 30;

    tmp3 ^= tmp6 << 11;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 87;
    tmp10 = tmp10 ^ (tmp10 << 11);

    *res = tmp3 ^ tmp10;

    *res = *res << 30 >> 30;
}

inline void gfmul99(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 29) | (tmp3 >> 99); 
    tmp3 = tmp3 << 29 >> 29;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 93;
    tmp11 = tmp6 >> 96;
    tmp12 = tmp6 >> 98;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 29 >> 29;
}

inline void gfmul100(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 28) | (tmp3 >> 100); 
    tmp3 = tmp3 << 28 >> 28;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 28 >> 28;
}

inline void gfmul101(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 27) | (tmp3 >> 101); 
    tmp3 = tmp3 << 27 >> 27;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp11 = tmp6 >> 95;
    tmp12 = tmp6 >> 100;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 27 >> 27;
}

inline void gfmul102(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 26) | (tmp3 >> 102); 
    tmp3 = tmp3 << 26 >> 26;

    tmp3 ^= tmp6 << 29;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 73;
    tmp10 = tmp10 ^ (tmp10 << 29);

    *res = tmp3 ^ tmp10;

    *res = *res << 26 >> 26;
}

inline void gfmul103(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 25) | (tmp3 >> 103); 
    tmp3 = tmp3 << 25 >> 25;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 94;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 25 >> 25;
}

inline void gfmul104(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 24) | (tmp3 >> 104); 
    tmp3 = tmp3 << 24 >> 24;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 100;
    tmp11 = tmp6 >> 101;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 24 >> 24;
}

inline void gfmul105(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 23) | (tmp3 >> 105); 
    tmp3 = tmp3 << 23 >> 23;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 23 >> 23;
}

inline void gfmul106(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 22) | (tmp3 >> 106); 
    tmp3 = tmp3 << 22 >> 22;

    tmp3 ^= tmp6 << 15;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 15);

    *res = tmp3 ^ tmp10;

    *res = *res << 22 >> 22;
}

inline void gfmul107(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 21) | (tmp3 >> 107); 
    tmp3 = tmp3 << 21 >> 21;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 98;
    tmp11 = tmp6 >> 100;
    tmp12 = tmp6 >> 103;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 9) ^ (tmp10 << 7) ^ (tmp10 << 4);

    *res = tmp3 ^ tmp10;

    *res = *res << 21 >> 21;
}

inline void gfmul108(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 20) | (tmp3 >> 108); 
    tmp3 = tmp3 << 20 >> 20;

    tmp3 ^= tmp6 << 17;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 91;
    tmp10 = tmp10 ^ (tmp10 << 17);

    *res = tmp3 ^ tmp10;

    *res = *res << 20 >> 20;
}

inline void gfmul109(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 19) | (tmp3 >> 109); 
    tmp3 = tmp3 << 19 >> 19;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp11 = tmp6 >> 105;
    tmp12 = tmp6 >> 107;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 19 >> 19;
}

inline void gfmul110(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 18) | (tmp3 >> 110); 
    tmp3 = tmp3 << 18 >> 18;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 77;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 18 >> 18;
}

inline void gfmul111(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 17) | (tmp3 >> 111); 
    tmp3 = tmp3 << 17 >> 17;

    tmp3 ^= tmp6 << 10;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 101;
    tmp10 = tmp10 ^ (tmp10 << 10);

    *res = tmp3 ^ tmp10;

    *res = *res << 17 >> 17;
}

inline void gfmul112(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 16) | (tmp3 >> 112); 
    tmp3 = tmp3 << 16 >> 16;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 109;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 4) ^ (tmp10 << 3);

    *res = tmp3 ^ tmp10;

    *res = *res << 16 >> 16;
}

inline void gfmul113(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 15) | (tmp3 >> 113); 
    tmp3 = tmp3 << 15 >> 15;

    tmp3 ^= tmp6 << 9;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 104;
    tmp10 = tmp10 ^ (tmp10 << 9);

    *res = tmp3 ^ tmp10;

    *res = *res << 15 >> 15;
}

inline void gfmul114(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 14) | (tmp3 >> 114); 
    tmp3 = tmp3 << 14 >> 14;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 109;
    tmp11 = tmp6 >> 111;
    tmp12 = tmp6 >> 112;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 3) ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 14 >> 14;
}

inline void gfmul115(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 13) | (tmp3 >> 115); 
    tmp3 = tmp3 << 13 >> 13;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 107;
    tmp11 = tmp6 >> 108;
    tmp12 = tmp6 >> 110;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 8) ^ (tmp10 << 7) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 13 >> 13;
}

inline void gfmul116(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 12) | (tmp3 >> 116); 
    tmp3 = tmp3 << 12 >> 12;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 114;
    tmp12 = tmp6 >> 115;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 12 >> 12;
}

inline void gfmul117(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 11) | (tmp3 >> 117); 
    tmp3 = tmp3 << 11 >> 11;

    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 112;
    tmp11 = tmp6 >> 115;
    tmp12 = tmp6 >> 116;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 5) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 11 >> 11;
}

inline void gfmul118(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 10) | (tmp3 >> 118); 
    tmp3 = tmp3 << 10 >> 10;

    tmp3 ^= tmp6 << 33;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 85;
    tmp10 = tmp10 ^ (tmp10 << 33);

    *res = tmp3 ^ tmp10;

    *res = *res << 10 >> 10;
}

inline void gfmul119(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 9) | (tmp3 >> 119); 
    tmp3 = tmp3 << 9 >> 9;

    tmp3 ^= tmp6 << 8;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 111;
    tmp10 = tmp10 ^ (tmp10 << 8);

    *res = tmp3 ^ tmp10;

    *res = *res << 9 >> 9;
}

inline void gfmul120(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 8) | (tmp3 >> 120); 
    tmp3 = tmp3 << 8 >> 8;

    tmp3 ^= tmp6 << 4;
    tmp3 ^= tmp6 << 3;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 117;
    tmp12 = tmp6 >> 119;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 4) ^ (tmp10 << 3) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 8 >> 8;
}

inline void gfmul121(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 7) | (tmp3 >> 121); 
    tmp3 = tmp3 << 7 >> 7;

    tmp3 ^= tmp6 << 18;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 103;
    tmp10 = tmp10 ^ (tmp10 << 18);

    *res = tmp3 ^ tmp10;

    *res = *res << 7 >> 7;
}

inline void gfmul122(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 6) | (tmp3 >> 122); 
    tmp3 = tmp3 << 6 >> 6;

    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 116;
    tmp11 = tmp6 >> 120;
    tmp12 = tmp6 >> 121;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 6) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 6 >> 6;
}

inline void gfmul123(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 5) | (tmp3 >> 123); 
    tmp3 = tmp3 << 5 >> 5;

    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 121;
    tmp10 = tmp10 ^ (tmp10 << 2);

    *res = tmp3 ^ tmp10;

    *res = *res << 5 >> 5;
}

inline void gfmul124(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 4) | (tmp3 >> 124); 
    tmp3 = tmp3 << 4 >> 4;

    tmp3 ^= tmp6 << 19;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 19);

    *res = tmp3 ^ tmp10;

    *res = *res << 4 >> 4;
}

inline void gfmul125(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 3) | (tmp3 >> 125); 
    tmp3 = tmp3 << 3 >> 3;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 6;
    tmp3 ^= tmp6 << 5;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 118;
    tmp11 = tmp6 >> 119;
    tmp12 = tmp6 >> 120;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 6) ^ (tmp10 << 5);

    *res = tmp3 ^ tmp10;

    *res = *res << 3 >> 3;
}

inline void gfmul126(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 2) | (tmp3 >> 126); 
    tmp3 = tmp3 << 2 >> 2;

    tmp3 ^= tmp6 << 21;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 105;
    tmp10 = tmp10 ^ (tmp10 << 21);

    *res = tmp3 ^ tmp10;

    *res = *res << 2 >> 2;
}

inline void gfmul127(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp6 = (tmp6 << 1) | (tmp3 >> 127); 
    tmp3 = tmp3 << 1 >> 1;

    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 126;
    tmp10 = tmp10 ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

    *res = *res << 1 >> 1;
}

inline void gfmul128(__int128 a, __int128 b, __int128 *res){
    __int128 tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    uint64_t a_lo = a, a_hi = a >> 64;
    uint64_t b_lo = b, b_hi = b >> 64;
    uint64_t tmp3_lo, tmp3_hi, tmp4_lo, tmp4_hi, 
             tmp5_lo, tmp5_hi, tmp6_lo, tmp6_hi;

    __asm__ __volatile__ (
        "mv t0, %8\n"             
        "mv t1, %9\n"             
        "mv t2, %10\n"             
        "mv t3, %11\n"             

        "clmul   %0, t0, t1\n"   
        "clmulh  %1, t0, t1\n" 
        "clmul   %2, t2, t1\n" 
        "clmulh  %3, t2, t1\n" 
        "clmul   %4, t0, t3\n" 
        "clmulh  %5, t0, t3\n" 
        "clmul   %6, t2, t3\n" 
        "clmulh  %7, t2, t3\n" 

        : "=r" (tmp3_lo), "=r" (tmp3_hi), 
          "=r" (tmp4_lo), "=r" (tmp4_hi), 
          "=r" (tmp5_lo), "=r" (tmp5_hi), 
          "=r" (tmp6_lo), "=r" (tmp6_hi)
        : "r" (a_lo), "r" (b_lo), "r" (a_hi), "r" (b_hi)
        : "t0", "t1", "t2", "t3"
    );

    tmp3 = ((__int128)tmp3_hi << 64) | tmp3_lo;
    tmp4 = ((__int128)tmp4_hi << 64) | tmp4_lo;
    tmp5 = ((__int128)tmp5_hi << 64) | tmp5_lo;
    tmp6 = ((__int128)tmp6_hi << 64) | tmp6_lo;

    tmp4 = tmp4 ^ tmp5;
    tmp5 = tmp4 << 64;
    tmp4 = tmp4 >> 64;
    tmp3 = tmp3 ^ tmp5;
    tmp6 = tmp6 ^ tmp4;

    tmp3 ^= tmp6 << 7;
    tmp3 ^= tmp6 << 2;
    tmp3 ^= tmp6 << 1;
    tmp3 ^= tmp6;

    tmp10 = tmp6 >> 121;
    tmp11 = tmp6 >> 126;
    tmp12 = tmp6 >> 127;
    tmp10 ^= tmp11;
    tmp10 ^= tmp12;
    tmp10 = tmp10 ^ (tmp10 << 7) ^ (tmp10 << 2) ^ (tmp10 << 1);

    *res = tmp3 ^ tmp10;

}
#endif

