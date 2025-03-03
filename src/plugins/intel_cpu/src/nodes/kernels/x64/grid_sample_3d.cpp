// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample.hpp"
#include <immintrin.h>
#include "../../../emitters/plugin/x64/debug_capabilities.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(GridSamplesKernelExecArgs, field)


template <x64::cpu_isa_t isa>
GridSample3DSimpleKernel<isa>::GridSample3DSimpleKernel(const GridSampleKernelConfParams& jcp)
    : GridSampleKernelBase(jit_name(), jcp, isa) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void GridSample3DSimpleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        OPENVINO_THROW("Could not create GridSample3DSimpleKernel kernel. Error code: ", std::to_string(code));
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void GridSample3DSimpleKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    process();

    registersPool.reset();
    this->postamble();
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DSimpleKernel<isa>::process() {
    auto regWorkAmount = getReg64();
    auto regSrc = getReg64();
    auto regDst = getReg64();
    auto vTmp64 = getVmm();
    
    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

    Xbyak::Label lLoop512, lTail256, lTail128, lTail64, lEnd;
    L(lLoop512);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail256, T_NEAR);

        uni_vmovups(vTmp64, ptr[regSrc]);
        uni_vmovups(ptr[regDst], vTmp64);

        sub(regWorkAmount, vlen);
        add(regSrc, vlen);
        add(regDst, vlen);

        jmp(lLoop512, T_NEAR);
    }

    L(lTail256);
    cmp(regWorkAmount, dataElPerVec/2);
    jl(lTail128, T_NEAR);
    vmovdqu64(vTmp64, ptr[regSrc]);
    vmovdqu64(ptr[regDst], vTmp64);
    sub(regWorkAmount, vlen/2);
    add(regSrc, vlen/2);
    add(regDst, vlen/2);

    L(lTail128);
    cmp(regWorkAmount, dataElPerVec/4);
    jl(lTail64, T_NEAR);
    vmovdqu64(vTmp64, ptr[regSrc]);
    vmovdqu64(ptr[regDst], vTmp64);
    sub(regWorkAmount, vlen/4);
    add(regSrc, vlen/4);
    add(regDst, vlen/4);

    L(lTail64);
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);
    mov(al, byte[regSrc]);
    mov(byte[regDst], al);
    dec(regWorkAmount);
    inc(regSrc);
    inc(regDst);
    jmp(lTail64, T_NEAR);

    L(lEnd);
}
template class GridSample3DSimpleKernel<x64::avx512_core>;

template <x64::cpu_isa_t isa>
GridSample3DKernel<isa>::GridSample3DKernel(const GridSampleKernelConfParams& jcp)
    : GridSampleKernelBase(jit_name(), jcp, isa) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void GridSample3DKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        OPENVINO_THROW("Could not create GridSample kernel. Error code: ", std::to_string(code));
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void GridSample3DKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    regSrc = getReg64();
    regGrid = getReg64();
    regDst = getReg64();
    regSrcChannelStepB = getReg64();
    regDstChannelStepB = getReg64();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void GridSample3DKernel<x64::avx512_core>::initVectors() {
    auto rAux = getReg64();
    Xbyak::Reg32 r32Aux(rAux.getIdx());

    if (jcp.dynamicShapes) {
        regChannelNum = getReg64();
        mov(regChannelNum, ptr[regParams + GET_OFF(channelsNum)]);
    }
    kTailMask = getMask();

    vSrcDepthF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
    uni_vpbroadcastd(vSrcDepthF, ptr[rAux]);

    vSrcWidthF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vpbroadcastd(vSrcWidthF, ptr[rAux]);

    vSrcHeightF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
    uni_vpbroadcastd(vSrcHeightF, ptr[rAux]);

    vZeros = getVmm();
    uni_vpxor(vZeros, vZeros, vZeros);

    if (one_of(jcp.interpolationMode, GridSampleInterpolationMode::BILINEAR)) {
        vOnesF = getVmm();
        mov(r32Aux, 0x3f800000);  // 1.f
        vpbroadcastd(vOnesF, r32Aux);
    }

    if (jcp.alignCorners) {
        vDDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(dDenormCoefF)]);
        uni_vpbroadcastd(vDDenormCoefF, ptr[rAux]);

        vWDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vpbroadcastd(vWDenormCoefF, ptr[rAux]);

        vHDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vpbroadcastd(vHDenormCoefF, ptr[rAux]);
    } else {
        vHalfF = getVmm();
        mov(r32Aux, 0x3f000000);  // 0.5f
        vpbroadcastd(vHalfF, r32Aux);
    }
    // xyz.xyz.xyz.xyz.xyz.x
    static const unsigned gridPermMask[16] = {0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14, };
    mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
    vGridPermMask = getVmm();
    uni_vmovups(vGridPermMask, ptr[rAux]);

    static const unsigned xMask[16] = {0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13};
    mov(rAux, reinterpret_cast<uintptr_t>(xMask));
    vXMask = getVmm();
    uni_vmovups(vXMask, ptr[rAux]);

    static const unsigned yMask[16] = {6, 7, 8, 9, 10, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14};
    mov(rAux, reinterpret_cast<uintptr_t>(yMask));
    vYMask = getVmm();
    uni_vmovups(vYMask, ptr[rAux]);


    static const unsigned zMask[16] = {11, 12, 13, 14, 15, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15};
    mov(rAux, reinterpret_cast<uintptr_t>(zMask));
    vZMask = getVmm();
    uni_vmovups(vZMask, ptr[rAux]);


    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        vDataTypeSizeB = getVmm();
        mov(rAux, dataTypeSize);
        vpbroadcastd(vDataTypeSizeB, r32Aux);
        vSrcWidthB = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthB)]);
        uni_vpbroadcastd(vSrcWidthB, ptr[rAux]);
        vSrcWidthHeightB = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthHeightB)]);
        uni_vpbroadcastd(vSrcWidthHeightB, ptr[rAux]);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        vSrcDepthSub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthSub1F)]);
        uni_vpbroadcastd(vSrcDepthSub1F, ptr[rAux]);
        vSrcHeightSub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
        uni_vpbroadcastd(vSrcHeightSub1F, ptr[rAux]);
        vSrcWidthSub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
        uni_vpbroadcastd(vSrcWidthSub1F, ptr[rAux]);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        vSrcDepthMul2F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthMul2F)]);
        uni_vpbroadcastd(vSrcDepthMul2F, ptr[rAux]);
        vSrcHeightMul2F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        uni_vpbroadcastd(vSrcHeightMul2F, ptr[rAux]);
        vSrcWidthMul2F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        uni_vpbroadcastd(vSrcWidthMul2F, ptr[rAux]);
        vSrcDepthMul2Sub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthMul2Sub1F)]);
        uni_vpbroadcastd(vSrcDepthMul2Sub1F, ptr[rAux]);
        vSrcHeightMul2Sub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        uni_vpbroadcastd(vSrcHeightMul2Sub1F, ptr[rAux]);
        vSrcWidthMul2Sub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        uni_vpbroadcastd(vSrcWidthMul2Sub1F, ptr[rAux]);
        if (jcp.alignCorners) {
            vAbsMask = getVmm();
            mov(r32Aux, 0x7fffffff);
            vpbroadcastd(vAbsMask, r32Aux);
        }
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::initVectors() {
    auto rAux = getReg64();

    vSrcWidthF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vmovups(vSrcWidthF, ptr[rAux]);

    if (one_of(jcp.interpolationMode, GridSampleInterpolationMode::BILINEAR, GridSampleInterpolationMode::NEAREST) ||
        (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC &&
         (jcp.paddingMode == GridSamplePaddingMode::REFLECTION ||
          (jcp.paddingMode == GridSamplePaddingMode::BORDER && !jcp.alignCorners) ||
          jcp.paddingMode == GridSamplePaddingMode::ZEROS))) {
        vSrcHeightF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vmovups(vSrcHeightF, ptr[rAux]);
        vSrcDepthF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
        uni_vmovups(vSrcDepthF, ptr[rAux]);
    }

    if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC &&
        jcp.paddingMode == GridSamplePaddingMode::BORDER && jcp.alignCorners) {
        vDDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(dDenormCoefF)]);
        uni_vmovups(vDDenormCoefF, ptr[rAux]);
        vHDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vHDenormCoefF, ptr[rAux]);
    }

    if (jcp.interpolationMode != GridSampleInterpolationMode::BICUBIC) {
        if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::ZEROS) &&
            ((isa == x64::avx2 && jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) ||
             one_of(isa, x64::avx, x64::sse41))) {
            vZeros = getVmm();
            uni_vpxor(vZeros, vZeros, vZeros);
        }

        if (jcp.alignCorners) {
            mov(rAux, ptr[regParams + GET_OFF(dDenormCoefF)]);
            vDDenormCoefF = getVmm();
            uni_vmovups(vDDenormCoefF, ptr[rAux]);
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            vWDenormCoefF = getVmm();
            uni_vmovups(vWDenormCoefF, ptr[rAux]);
            if (!(jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR &&
                  jcp.paddingMode == GridSamplePaddingMode::ZEROS)) {
                mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
                vHDenormCoefF = getVmm();
                uni_vmovups(vHDenormCoefF, ptr[rAux]);
            }
        } else {
            static const float halfArr[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            mov(rAux, reinterpret_cast<uintptr_t>(halfArr));
            vHalfF = getVmm();
            uni_vmovups(vHalfF, ptr[rAux]);
        }

        if (isa == x64::avx2 && jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) {
            static const unsigned gridPermMask[8] = {0, 3, 1, 4, 2, 5, 6, 7};
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            vGridPermMask = getVmm();
            uni_vmovups(vGridPermMask, ptr[rAux]);
        }
    }

    if (jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR &&
         jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
        static const float onesArr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
        vOnesF = getVmm();
        uni_vmovups(vOnesF, ptr[rAux]);
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::process() {
    regWorkAmount = getReg64();

    // Batch loop
    Xbyak::Label lBatchLoop, lEnd;
    RegistersPool::Reg<Xbyak::Reg64> regBatch;

    for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
        if (jcp.dynamicBatch) {
            regBatch = getReg64();
            mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);

            L(lBatchLoop);
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);
        }

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

        spatialLoop();

        if (jcp.dynamicShapes) {
            add(regSrc, ptr[regParams + GET_OFF(srcBatchStepB)]);
        } else {
            add(regSrc, jcp.srcBatchStepB);
        }
        add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
        add(regDst, ptr[regParams + GET_OFF(dstBatchStepB)]);

        if (jcp.dynamicBatch) {
            dec(regBatch);
            jmp(lBatchLoop, T_NEAR);
            L(lEnd);
        }
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::spatialLoop() {
    auto vDCoord = getVmm();
    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);
        getCoordinates(vDCoord, vHCoord, vWCoord);
        denormalizeRawCoordinates(vWCoord, vHCoord, vDCoord);
        interpolation(vWCoord, vHCoord, vDCoord);
        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    vDCoord.release();
    vHCoord.release();
    vWCoord.release();
    tail();
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::interpolation(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord, bool tail) {
    if (jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR) {
        bilinearInterpolation(vWCoord, vHCoord, vDCoord, tail);
    } else if (jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) {
        nearestInterpolation(vWCoord, vHCoord, vDCoord, tail);
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::tail() {
    Xbyak::Label lEnd;
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);

    auto vDCoord = getVmm();
    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    getTailCoordinates(vDCoord, vHCoord, vWCoord);
    denormalizeRawCoordinates(vWCoord, vHCoord, vDCoord);
    interpolation(vWCoord, vHCoord, vDCoord, true);

    if (dataTypeSize > 1)
        sal(regWorkAmount, dataTypeShift);  // Multiply by source data type size.
    add(regDst, regWorkAmount);

    L(lEnd);
}

template <>
void GridSample3DKernel<x64::avx512_core>::getCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto kMask0 = getMask();
    const __mmask16 keys[] = {0x07C0, 0x07E0, 0x3E0, 0xF800, 0xFC00};
    auto r32Aux = getReg32();

    vpermd(vWCoord, vGridPermMask, ptr[regGrid]);       // Permute from xyz.xyz.xyz.xyz.xyz.x to XXXX.XXYY.YYYZ.ZZZZ
    vpermd(vHCoord, vYMask, vWCoord);
    vpermd(vDCoord, vZMask, vWCoord);

    add(regGrid, vlen);

    auto vAux = getVmm();
    vmovdqu32(vAux, ptr[regGrid]);      // yz.xyz.xyz.xyz.xyz.xy 

    mov(r32Aux, keys[0]);
    kmovw(kMask0, r32Aux);
    vpermd(vWCoord | kMask0, vXMask, vAux);

    mov(r32Aux, keys[1]);
    kmovw(kMask0, r32Aux);
    vpermd(vHCoord | kMask0, vYMask, vAux);

    mov(r32Aux, keys[2]);
    kmovw(kMask0, r32Aux);
    vpermd(vDCoord | kMask0, vZMask, vAux);

    add(regGrid, vlen);

    vmovdqu32(vAux, ptr[regGrid]);      // z.xyz.xyz.xyz.xyz.xyz

    mov(r32Aux, keys[3]);
    kmovw(kMask0, r32Aux);
    vpermd(vWCoord | kMask0, vXMask, vAux);
    vpermd(vHCoord | kMask0, vYMask, vAux);

    mov(r32Aux, keys[4]);
    kmovw(kMask0, r32Aux);
    vpermd(vDCoord | kMask0, vZMask, vAux);

    add(regGrid, vlen);
}

template <>
void GridSample3DKernel<x64::avx2>::getCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        auto rAux = getReg64();
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    vpermd(vWCoord, vPermMask, ptr[regGrid]);           // Permute to XXXX.YYYY
    vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011);  // Extract Y component

    add(regGrid, vlen);

    vpermd(vAux, vPermMask, ptr[regGrid]);           // Permute to XXXX.YYYY
    vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);  // Extract X component
    vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);  // Extract Y component

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa>  // Works for AVX, SSE41
void GridSample3DKernel<isa>::getCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
    Xbyak::Xmm xmmAux(vAux.getIdx());
    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

    uni_vmovups(xmmWCoord, ptr[regGrid]);
    uni_vpshufd(xmmWCoord, xmmWCoord, 0xD8);
    shufpd(xmmHCoord, xmmWCoord, 0x2);

    add(regGrid, xmmVlen);

    uni_vmovups(xmmAux, ptr[regGrid]);
    uni_vpshufd(xmmAux, xmmAux, 0xD8);
    shufpd(xmmWCoord, xmmAux, 0x0);
    shufpd(xmmHCoord, xmmAux, 0x3);

    add(regGrid, xmmVlen);

    if (isa == x64::avx) {
        Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
        Xbyak::Ymm ymmHCoord(vHCoord.getIdx());

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);

        // Here is movups + pshufd instead of vpshufd for two reasons:
        // 1. vpshufd zeroes the rest ov YMM.
        // 2. pshufd does not work with not aligned address.
        movups(xmmWCoord, ptr[regGrid]);
        pshufd(xmmWCoord, xmmWCoord, 0xD8);
        shufpd(xmmHCoord, xmmWCoord, 0x2);

        add(regGrid, xmmVlen);

        uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
        shufpd(xmmWCoord, xmmAux, 0x0);
        shufpd(xmmHCoord, xmmAux, 0x3);

        add(regGrid, xmmVlen);

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
    }
}

template <>
void GridSample3DKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lEnd;

    auto rAux = getReg64();

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1);  // Multiply by 2
    add(rAux, regWorkAmount);  // Add original value to get rAux * 3
    cmp(regWorkAmount, dataElPerVec / 2);

    vpermd(vWCoord, vGridPermMask, ptr[regGrid]);       // Permute from xyz.xyz.xyz.xyz.xyz.x to XXXX.XXYY.YYYZ.ZZZZ
    vpermd(vHCoord, vYMask, vWCoord);
    vpermd(vDCoord, vZMask, vWCoord);
    add(regGrid, vlen);

    sub(rAux, dataElPerVec);
    cmp(rAux, 0);
    jle(lEnd, T_NEAR);

    const __mmask16 keys[] = {0x07C0, 0x07E0, 0x3E0, 0xF800, 0xFC00};
    auto kMask0 = getMask();
    auto r32Aux = getReg32();

    auto vAux = getVmm();
    vmovdqu32(vAux, ptr[regGrid]);      // yz.xyz.xyz.xyz.xyz.xy 

    mov(r32Aux, keys[0]);
    kmovw(kMask0, r32Aux);
    vpermd(vWCoord | kMask0, vXMask, vAux);

    mov(r32Aux, keys[1]);
    kmovw(kMask0, r32Aux);
    vpermd(vHCoord | kMask0, vYMask, vAux);

    mov(r32Aux, keys[2]);
    kmovw(kMask0, r32Aux);
    vpermd(vDCoord | kMask0, vZMask, vAux);

    add(regGrid, vlen);

    sub(rAux, dataElPerVec);
    cmp(rAux, 0);
    jle(lEnd, T_NEAR);

    vmovdqu32(vAux, ptr[regGrid]);      // z.xyz.xyz.xyz.xyz.xyz

    mov(r32Aux, keys[3]);
    kmovw(kMask0, r32Aux);
    vpermd(vWCoord | kMask0, vXMask, vAux);
    vpermd(vHCoord | kMask0, vYMask, vAux);

    mov(r32Aux, keys[4]);
    kmovw(kMask0, r32Aux);
    vpermd(vDCoord | kMask0, vZMask, vAux);

    add(regGrid, vlen);

    L(lEnd);

    fillRestWorkMask(kTailMask, regWorkAmount);
}

template <>
void GridSample3DKernel<x64::avx2>::getTailCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;

    auto rAux = getReg64();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1);  // multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        vpermd(vWCoord, vPermMask, ptr[regGrid]);           // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011);  // Extract Y component

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        auto vAux = getVmm();
        load(vAux, ptr[regGrid], rAux, dataTypeSize);
        vpermd(vAux, vPermMask, vAux);
        vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);  // Extract X component
        vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);  // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
        vpermd(vWCoord, vPermMask, vWCoord);                // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011);  // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift);  // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <>
void GridSample3DKernel<x64::avx>::getTailCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lLoop2End, lEnd;

    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());

    auto rGridRest = getReg64();
    mov(rGridRest, regWorkAmount);
    sal(rGridRest, 0x1);  // multiply by gridShape[3] == 2

    for (size_t i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lEnd, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    cmp(rGridRest, 0);
    jle(lEnd, T_NEAR);

    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    for (size_t i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lLoop2End, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    L(lLoop2End);
    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    L(lEnd);
}

template <>
void GridSample3DKernel<x64::sse41>::getTailCoordinates(const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lHShuf, lGridShift, lEnd;
    auto rAux = getReg64();

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1);  // Multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        // Here is movups + pshufd instead of pshufd due to
        // pshufd does not work with not aligned address.
        movups(vWCoord, ptr[regGrid]);
        pshufd(vWCoord, vWCoord, 0B11011000);
        shufpd(vHCoord, vWCoord, 0B00000010);

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lHShuf, T_NEAR);

        auto vAux = getVmm();
        load(vAux, ptr[regGrid], rAux, dataTypeSize);
        pshufd(vAux, vAux, 0B11011000);
        shufpd(vWCoord, vAux, 0x0);         // Extract X component
        shufpd(vHCoord, vAux, 0B00000011);  // Extract Y component

        jmp(lGridShift, T_NEAR);
        L(lHShuf);
        shufpd(vHCoord, vHCoord, 0B00000001);  // Extract Y component
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
        pshufd(vWCoord, vWCoord, 0B11011000);  // Extract X component
        shufpd(vHCoord, vWCoord, 0B00000010);  // Extract Y component
        shufpd(vHCoord, vHCoord, 0B00000001);
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift);  // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord) {
    if (jcp.alignCorners) {
        if (vDDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vDCoord, vDDenormCoefF, vDDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(dDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vDCoord, vAux, vAux);
        }
        
        if (vWDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vWDenormCoefF, vWDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }

        if (vHDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vHDenormCoefF, vHDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
    } else {
        Vmm vHalfTmp;
        RegistersPool::Reg<Vmm> halfHolder;
        if (vHalfF.isInitialized()) {
            vHalfTmp = vHalfF;
        } else {
            auto rAux = getReg64();
            halfHolder = getVmm();
            vHalfTmp = halfHolder;
            static const float halfValues[x64::cpu_isa_traits<x64::avx512_core>::vlen / sizeof(float)] =
                {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
            uni_vmovups(vHalfTmp, ptr[rAux]);
        }

        if (vSrcWidthF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vWCoord, vHalfTmp, vHalfTmp);
        
        if (vSrcHeightF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vHCoord, vHalfTmp, vHalfTmp);

        if (vSrcDepthF.isInitialized()) {
            uni_vfmadd132ps(vDCoord, vSrcDepthF, vSrcDepthF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vDCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vDCoord, vHalfTmp, vHalfTmp);
    }
}

template <>
void GridSample3DKernel<x64::avx512_core>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
    vcmpps(kDst, vCoord, vSrcWidthF, CMP_LT_PS);     // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS);  // vCoord >= vZeros
}

template <>
void GridSample3DKernel<x64::avx512_core>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
    vcmpps(kDst | kMaskW, vCoord, vSrcHeightF, CMP_LT_PS);  // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS);         // vCoord >= vZeros
}

template <>
void GridSample3DKernel<x64::avx512_core>::zerosPaddingD(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskWH) {
    vcmpps(kDst | kMaskWH, vCoord, vSrcDepthF, CMP_LT_PS);  // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS);         // vCoord >= vZeros
}

template <>
void GridSample3DKernel<x64::avx512_core>::zerosPadding(const Vmask& kDst, const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);
    zerosPaddingD(kDst, vDCoord, kDst);
}

template <>
void GridSample3DKernel<x64::sse41>::zerosPaddingW(const Vmask& kDst, const Vmm& vWCoord) {
    auto vAux = getVmm();

    if (vSrcWidthF.isInitialized()) {
        uni_vcmpps(vAux, vWCoord, vSrcWidthF, CMP_LT_PS);  // vWCoord < vSrcWidthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vcmpps(vAux, vWCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
    }

    uni_vpxor(kDst, kDst, kDst);
    uni_vcmpps(kDst, kDst, vWCoord, CMP_LE_PS);  // vWCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);                 // vZeros <= vWCoord < vSrcWidthF
}

template <>
void GridSample3DKernel<x64::sse41>::zerosPaddingH(const Vmask& kDst, const Vmm& vHCoord, const Vmask& kMaskW) {
    auto vAux = getVmm();

    if (vSrcHeightF.isInitialized()) {
        uni_vcmpps(vAux, vHCoord, vSrcHeightF, CMP_LT_PS);  // vHCoord < vSrcHeightF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vcmpps(vAux, vHCoord, ptr[rAux], CMP_LT_PS);  // vHCoord < vSrcHeightF
    }

    uni_vmovups(kDst, kMaskW);
    uni_vpand(kDst, kDst, vAux);  // vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vHCoord, CMP_LE_PS);  // vHCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);                 // vZeros <= vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
}

template <>
void GridSample3DKernel<x64::sse41>::zerosPaddingD(const Vmask& kDst, const Vmm& vDCoord, const Vmask& kMaskWH) {
    auto vAux = getVmm();

    if (vSrcDepthF.isInitialized()) {
        uni_vcmpps(vAux, vDCoord, vSrcDepthF, CMP_LT_PS);  // vDCoord < vSrcDepthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
        uni_vcmpps(vAux, vDCoord, ptr[rAux], CMP_LT_PS);  // vDCoord < vSrcDepthF
    }

    uni_vmovups(kDst, kMaskWH);
    uni_vpand(kDst, kDst, vAux);  // vDCoord < vSrcDepthF && vZeros <= vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vDCoord, CMP_LE_PS);  // vDCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);                 // vZeros <= vDCoord < vSrcDepthF && vZeros <= vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
}

template <>
void GridSample3DKernel<x64::sse41>::zerosPadding(const Vmask& kDst, const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);
    zerosPaddingD(kDst, vDCoord, kDst);
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX
void GridSample3DKernel<isa>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }

    if (vSrcWidthF.isInitialized()) {
        uni_vcmpps(vAux, vCoord, vSrcWidthF, CMP_LT_PS);  // vWCoord < vSrcWidthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
    }

    uni_vcmpps(kDst, vZerosTmp, vCoord, CMP_LE_PS);  // vWCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);                    // vZeros <= vWCoord < vSrcWidthF
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX
void GridSample3DKernel<isa>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }

    if (vSrcHeightF.isInitialized()) {
        uni_vcmpps(vAux, vCoord, vSrcHeightF, CMP_LT_PS);  // vHCoord < vSrcHeightF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);  // vHCoord < vSrcHeightF
    }

    uni_vandps(kDst, kMaskW, vAux);
    uni_vcmpps(vAux, vZerosTmp, vCoord, CMP_LE_PS);  // vHCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX
void GridSample3DKernel<isa>::zerosPaddingD(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskWH) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }

    if (vSrcDepthF.isInitialized()) {
        uni_vcmpps(vAux, vCoord, vSrcDepthF, CMP_LT_PS);  // vHCoord < vSrcDepthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
        uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);  // vHCoord < vSrcDepthF
    }

    uni_vandps(kDst, kMaskWH, vAux);
    uni_vcmpps(vAux, vZerosTmp, vCoord, CMP_LE_PS);  // vHCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX
void GridSample3DKernel<isa>::zerosPadding(const Vmask& kDst, const Vmm& vDCoord, const Vmm& vHCoord, const Vmm& vWCoord) {
    bool releaseZeroVec = false;
    if (!vZeros.isInitialized()) {
        releaseZeroVec = true;
        vZeros = getVmm();
        uni_vpxor(vZeros, vZeros, vZeros);
    }

    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);
    zerosPaddingD(kDst, vDCoord, kDst);

    if (releaseZeroVec) {
        vZeros.release();
    }
}

template <>
void GridSample3DKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    vrangeps(vCoordDst,
             vCoordOrigin,
             dim == coord::w ? vSrcWidthSub1F : dim == coord::h ? vSrcHeightSub1F : vSrcDepthSub1F,
             0x0);                                // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);  // vWCoord < vZeros
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux = getReg64();
    auto vAux = getVmm();
    RegistersPool::Reg<Vmm> vAux1;

    Vmm vSub1F;
    if (dim == coord::w) {
        if (vSrcWidthSub1F.isInitialized()) {
            vSub1F = vSrcWidthSub1F;
        } else {
            vAux1 = getVmm();
            vSub1F = vAux1;
            mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
            uni_vmovups(vSub1F, ptr[rAux]);
        }
    } else if (dim == coord::h) {
        if (vSrcHeightSub1F.isInitialized()) {
            vSub1F = vSrcHeightSub1F;
        } else {
            vAux1 = getVmm();
            vSub1F = vAux1;
            mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
            uni_vmovups(vSub1F, ptr[rAux]);
        }
    } else if (dim == coord::d) {
        if (vSrcDepthSub1F.isInitialized()) {
            vSub1F = vSrcDepthSub1F;
        } else {
            vAux1 = getVmm();
            vSub1F = vAux1;
            mov(rAux, ptr[regParams + GET_OFF(srcDepthSub1F)]);
            uni_vmovups(vSub1F, ptr[rAux]);
        }
    }

    uni_vcmpps(vAux, vCoordOrigin, vSub1F, CMP_LE_PS);  // vCoord <= vUpperBound
    uni_vandps(vCoordDst, vCoordOrigin, vAux);
    uni_vandnps(vAux, vAux, vSub1F);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    if (vZeros.isInitialized()) {
        uni_vcmpps(vAux, vCoordDst, vZeros, 0x6);  // vCoord >= vZeros
    } else {
        if (isa == x64::sse41) {
            if (!vAux1.isInitialized()) {
                vAux1 = getVmm();
                vSub1F = vAux1;
            }
            uni_vpxor(vSub1F, vSub1F, vSub1F);
            uni_vcmpps(vAux, vCoordDst, vSub1F, 0x6);  // vCoord >= vZeros
        } else {
            uni_vpxor(vAux, vAux, vAux);
            uni_vcmpps(vAux, vCoordDst, vAux, 0x6);  // vCoord >= vZeros
        }
    }
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <>
void GridSample3DKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst,
                                                           const Vmm& vCoordOrigin,
                                                           const coord dim) {
    auto vAux = getVmm();
    auto kAux = getMask();
    const auto& vSrcDimMul2Sub1F = dim == coord::w ? vSrcWidthMul2Sub1F : 
                                   dim == coord::h ? vSrcHeightMul2Sub1F 
                                                   : vSrcDepthMul2Sub1F;

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vAbsMask);  // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3);                        // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F);  // abs(x) % D21

        // Check that the result does not exceed the divisor.
        vcmpps(kAux, vSrcDimMul2Sub1F, vCoordDst, CMP_LE_PS);
        uni_vmovups(vCoordDst | kAux, vZeros);
        vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
    } else {
        const auto& vSrcDimMul2F = dim == coord::w ? vSrcWidthMul2F : 
                                   dim == coord::h ? vSrcHeightMul2F 
                                                   : vSrcDepthMul2F;
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                    // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F);  // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F);   // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                    // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F);  // (x % D2 + D2) % D2

        // Check that the result does not exceed the divisor.
        vcmpps(kAux, vSrcDimMul2F, vCoordDst, CMP_LE_PS);
        uni_vmovups(vCoordDst | kAux, vZeros);
        vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, dim == coord::w ? vSrcWidthF : dim == coord::h ? vSrcHeightF : vSrcDepthF, vCoordDst, CMP_LE_PS);  // vCoordDst >= vSrcDimF
    uni_vmovups(vCoordDst | kAux, vAux);
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux = getReg64();
    auto vAux0 = getVmm();
    auto vAux1 = getVmm();

    // D2  = Dim * 2
    // D21 = (Dim - 1) * 2
    if (jcp.alignCorners) {
        // x' = abs(x) % D21 - D21
        static const unsigned absMask[8] =
            {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        if (isa == x64::sse41) {
            static const unsigned* absPtr = absMask + (reinterpret_cast<int64_t>(absMask) % 16) / sizeof(unsigned);
            mov(rAux, reinterpret_cast<uintptr_t>(absPtr));
        } else {
            mov(rAux, reinterpret_cast<uintptr_t>(absMask));
        }
        uni_vandps(vCoordDst, vCoordOrigin, ptr[rAux]);  // abs(x)

        Vmm vMul2Sub1;
        if (dim == coord::w) {
            if (vSrcWidthMul2Sub1F.isInitialized()) {
                vMul2Sub1 = vSrcWidthMul2Sub1F;
            } else {
                vMul2Sub1 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2Sub1F.isInitialized()) {
                vMul2Sub1 = vSrcHeightMul2Sub1F;
            } else {
                vMul2Sub1 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else {
            if (vSrcDepthMul2Sub1F.isInitialized()) {
                vMul2Sub1 = vSrcDepthMul2Sub1F;
            } else {
                vMul2Sub1 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcDepthMul2Sub1F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        }
        uni_vdivps(vAux0, vCoordDst, vMul2Sub1);
        uni_vroundps(vAux0, vAux0, 0x3);                // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2Sub1);  // abs(x) % D21

        // Check that the result does not exceed the divisor.
        uni_vcmpps(vAux0, vCoordDst, vMul2Sub1, CMP_LT_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);
        uni_vxorps(vAux0, vAux0, vAux0);
        uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);

        uni_vsubps(vAux0, vCoordDst, vMul2Sub1);  // abs(x) % D21 - D21
    } else {
        // x' = (x % D2 + D2) % D2 - D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        Vmm vMul2;
        if (dim == coord::w) {
            if (vSrcWidthMul2F.isInitialized()) {
                vMul2 = vSrcWidthMul2F;
            } else {
                vMul2 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2F.isInitialized()) {
                vMul2 = vSrcHeightMul2F;
            } else {
                vMul2 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else {
            if (vSrcDepthMul2F.isInitialized()) {
                vMul2 = vSrcDepthMul2F;
            } else {
                vMul2 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcDepthMul2F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        }
        uni_vdivps(vAux0, vCoordOrigin, vMul2);
        uni_vroundps(vAux0, vAux0, 0x3);            // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2);  // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vMul2);    // x % D2 + D2
        uni_vdivps(vAux0, vCoordDst, vMul2);
        uni_vroundps(vAux0, vAux0, 0x3);            // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2);  // (x % D2 + D2) % D2

        // Check that the result does not exceed the divisor.
        uni_vcmpps(vAux0, vCoordDst, vMul2, CMP_LT_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);
        uni_vxorps(vAux0, vAux0, vAux0);
        uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);

        if (dim == coord::w) {
            if (vSrcWidthMul2Sub1F.isInitialized()) {
                uni_vsubps(vAux0, vCoordDst, vSrcWidthMul2Sub1F);
            } else {
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2Sub1F.isInitialized()) {
                uni_vsubps(vAux0, vCoordDst, vSrcHeightMul2Sub1F);
            } else {
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
            }
        } else {
            if (vSrcDepthMul2Sub1F.isInitialized()) {
                uni_vsubps(vAux0, vCoordDst, vSrcDepthMul2Sub1F);
            } else {
                mov(rAux, ptr[regParams + GET_OFF(srcDepthMul2Sub1F)]);
                uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
            }
        }
    }

    if (dim == coord::w) {
        if (vSrcWidthF.isInitialized()) {
            uni_vcmpps(vAux1, vCoordDst, vSrcWidthF, CMP_LT_PS);  // vCoordDst < vUpperBound
        } else {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
            uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);  // vCoordDst < vUpperBound
        }
    } else if (dim == coord::h) {
        if (vSrcHeightF.isInitialized()) {
            uni_vcmpps(vAux1, vCoordDst, vSrcHeightF, CMP_LT_PS);  // vCoordDst < vUpperBound
        } else {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
            uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);  // vCoordDst < vUpperBound
        }
    } else {
        if (vSrcDepthF.isInitialized()) {
            uni_vcmpps(vAux1, vCoordDst, vSrcDepthF, CMP_LT_PS);  // vCoordDst < vUpperBound
        } else {
            mov(rAux, ptr[regParams + GET_OFF(srcDepthF)]);
            uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);  // vCoordDst < vUpperBound
        }
    }

    uni_vandps(vCoordDst, vCoordDst, vAux1);
    uni_vandnps(vAux1, vAux1, vAux0);
    uni_vsubps(vCoordDst, vCoordDst, vAux1);  // set -x' for vCoordDst >= Dim
}

template <x64::cpu_isa_t isa>  // Works for AVX512, AVX2, AVX, SSE41
void GridSample3DKernel<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord, bool tail) {
    const auto& vSrcShift = vWCoord;
    const auto& vAux = vHCoord;
    auto kGatherMask = getMask();
    auto kAuxMask = getMask();

    uni_vroundps(vDCoord, vDCoord, 0x0);  // Round near
    uni_vroundps(vWCoord, vWCoord, 0x0);  // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0);  // Round near

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        zerosPadding(kGatherMask, vDCoord, vHCoord, vWCoord);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
        borderPadding(vDCoord, vDCoord, coord::d);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, coord::w);
        reflectionPadding(vHCoord, vHCoord, coord::h);
        reflectionPadding(vDCoord, vDCoord, coord::d);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.channelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            if (isa == x64::avx512_core && tail)
                uni_kandd(kAuxMask, kTailMask, kGatherMask);
            else
                uni_kmovd(kAuxMask, kGatherMask);
        }

        if (!tail) {
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            uni_vmovups(ptr[rDstTmp], vAux);
        } else {
            if (isa == x64::avx512_core) {
                if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
                    uni_kmovd(kAuxMask, kTailMask);
                }
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, tail, zeroFill);
                uni_vmovups(ptr[rDstTmp] | Xbyak::Opmask(kTailMask.getIdx()), vAux);
            } else {
                memMovDD(rDstTmp, rSrcTmp, Vmm(kAuxMask.getIdx()), vSrcShift, regWorkAmount, useMask, zeroFill);
            }
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <>
void GridSample3DKernel<x64::avx512_core>::bilinearInterpolation2D0(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord, const Vmm& vDZ, bool tail) {
    auto vDX = getVmm();
    auto vDY = getVmm();
    auto shift00 = getVmm();
    auto shift01 = getVmm();
    auto shift10 = getVmm();
    auto shift11 = getVmm();
    auto vAux = getVmm();
    RegistersPool::Reg<Vmask> kMask00, kMask01, kMask10, kMask11;

    uni_vroundps(shift00, vWCoord, 0x1);  // Round floor x0
    uni_vroundps(shift10, vHCoord, 0x1);  // Round floor y0
    uni_vsubps(vDX, vWCoord, shift00);  
    uni_vsubps(vDY, vHCoord, shift01);
    uni_vaddps(shift01, shift00, vOnesF); // x1
    uni_vaddps(shift11, shift10, vOnesF); // y1  fisrt --x==0,y==1, second --number 0/1
    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        kMask00 = getMask();
        kMask01 = getMask();
        kMask10 = getMask();
        kMask11 = getMask();

        zerosPadding(kMask00, vDCoord, shift10, shift00);  // (z, y0; x0)
        zerosPadding(kMask01, vDCoord, shift10, shift01);  // (z, y0; x1)
        zerosPadding(kMask11, vDCoord, shift11, shift01);  // (z, y1; x1)
        zerosPadding(kMask10, vDCoord, shift11, shift00);  // (z, y1; x0)

        hwShiftPs2dq(shift00, shift10, shift00, vSrcWidthF);// y0,x0
        //TODO: add z offset z*vSrcWidthF*vSrcHeightF
        uni_vfmadd231ps(shift00, vDCoord, vSrcWidthHeightB);  //z, y0, x0, mask00

        uni_vpaddd(shift01, shift00, vDataTypeSizeB); // z, y0, x1, mask01
        uni_vpaddd(shift10, shift00, vSrcWidthB); //z, y1, x0, mask10
        uni_vpaddd(shift11, shift10, vDataTypeSizeB);//z, y1, x1, mask11
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(shift00, shift00, coord::w);
        borderPadding(shift01, shift01, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(shift00, shift00, coord::w);
        reflectionPadding(shift01, shift01, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (jcp.paddingMode == GridSamplePaddingMode::BORDER || jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, shift00, vSrcWidthF);
        hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
        hwShiftPs2dq(shift01, shift01, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto kAuxMask = getMask();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.channelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, 0);

            L(lChannelLoopBegin);
            cmp(rChannel, regChannelNum);
            jge(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill);  // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0);  // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
        }
        gatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX);  // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
        }
        gatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1);   // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux, vDX);  // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);
        uni_vmulps(vQ1, vQ1, vDZ); // Res = Res * dz
        
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3);  // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }
        RegPrinter::print<float>(*this, vQ1.reg, "Res0 out");

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            inc(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::bilinearInterpolation2D0(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord,
    const Vmm& vDZ, bool tail) {
    auto vWRound = getVmm();
    auto vHRound = getVmm();
    auto vDX = getVmm();
    auto vDY = getVmm();
    // const auto& vDX = vWCoord;
    // const auto& vDY = vHCoord;
    auto vAux = getVmm();
    Vmm shift00, shift01, shift10, shift11;
    RegistersPool::Reg<Vmm> shift10Holder, shift11Holder;
    // For ZEROS padding only.
    RegistersPool::Reg<Vmm> vMask00, vMask01, vMask10, vMask11;

    uni_vroundps(vWRound, vWCoord, 0x1);  // Round floor
    uni_vroundps(vHRound, vHCoord, 0x1);  // Round floor
    uni_vsubps(vDX, vDX, vWRound);
    uni_vsubps(vDY, vDY, vHRound);

    if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
        shift00 = vWRound;
        shift01 = vHRound;
        shift10Holder = getVmm();
        shift10 = shift10Holder;
        shift11Holder = getVmm();
        shift11 = shift11Holder;

        uni_vaddps(shift10, vWRound, vOnesF);
        uni_vaddps(shift11, vHRound, vOnesF);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        {
            auto rAux = getReg64();
            static const float onesArr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
            if (isa == x64::sse41) {
                static const float* onesPtr = onesArr + (reinterpret_cast<int64_t>(onesArr) % 16) / sizeof(float);
                mov(rAux, reinterpret_cast<uintptr_t>(onesPtr));
            } else {
                mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
            }
            uni_vmovups(vAux, ptr[rAux]);
        }
        shift00 = vWRound;
        shift10 = vHRound;
        vMask00 = getVmm();
        vMask01 = getVmm();
        vMask10 = getVmm();
        vMask11 = getVmm();

        uni_vaddps(vMask00, vWRound, vAux);
        uni_vaddps(vAux, vAux, vHRound);

        zerosPadding(vMask01, vDCoord, vHRound, vMask00);  // (y; x + 1)
        zerosPadding(vMask10, vDCoord, vAux, vWRound);     // (y + 1; x)
        zerosPadding(vMask11, vDCoord, vAux, vMask00);     // (y + 1; x + 1)
        zerosPadding(vMask00, vDCoord, vHRound, vWRound);  // (y; x)

        hwShiftPs2dq(shift00, vHRound, vWRound, vSrcWidthF);
        //TODO: add z offset z*vSrcWidthF*vSrcHeightF
        uni_vfmadd231ps(shift00, vDCoord, vSrcWidthHeightB);  
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(vWRound, vWRound, coord::w);
        borderPadding(vHRound, vHRound, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(vWRound, vWRound, coord::w);
        reflectionPadding(vHRound, vHRound, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::REFLECTION)) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWRound, vSrcWidthF);
        hwShiftPs2dq(vWRound, vHRound, vWRound, vSrcWidthF);
        hwShiftPs2dq(vHRound, vHRound, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto vGatherMask = getVmm();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    auto rTypeSize = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    mov(rTypeSize, ptr[regParams + GET_OFF(dataTypeSize)]);

    for (uint64_t ch = 0; ch < jcp.channelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS && isa == x64::avx2) {
            uni_vmovups(vGatherMask, vMask00);
        }
        gatherdd(vQ0,
                 rSrcTmp,
                 shift00,
                 (isa == x64::avx2 || !vMask00.isInitialized()) ? vGatherMask : vMask00,
                 useMask,
                 zeroFill);  // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ0, vDX, vQ0);  // q0 = -(v00 - dx * v00)
        } else {
            uni_vmulps(vGatherMask, vQ0, vDX);
            uni_vsubps(vQ0, vQ0, vGatherMask);
        }

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpaddd(shift10, shift00, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask01);
        }
        gatherdd(vAux,
                 rSrcTmp,
                 jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift01 : shift10,
                 (isa == x64::avx2 || !vMask01.isInitialized()) ? vGatherMask : vMask01,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        if (isa == x64::avx2) {
            uni_vfmsub231ps(vQ0, vAux, vDX);  // q0 = -q0 + dx * v01
        } else {
            uni_vmulps(vAux, vAux, vDX);
            uni_vaddps(vQ0, vQ0, vAux);
        }

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            {
                auto rSrcWidth = getReg64();
                mov(rSrcWidth, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpaddd(shift10, shift10, ptr[rSrcWidth]);
            }
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask11);
        }
        gatherdd(vAux,
                 rSrcTmp,
                 jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift11 : shift10,
                 (isa == x64::avx2 || !vMask11.isInitialized()) ? vGatherMask : vMask11,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpsubd(shift10, shift10, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask10);
        }
        gatherdd(vQ1,
                 rSrcTmp,
                 shift10,
                 (isa == x64::avx2 || !vMask10.isInitialized()) ? vGatherMask : vMask10,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        // q1 = -(v10 - dx * v10)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ1, vDX, vQ1);
        } else {
            uni_vmulps(vGatherMask, vQ1, vDX);
            if (isa == x64::avx) {
                uni_vsubps(vQ1, vGatherMask, vQ1);
            } else {
                uni_vsubps(vGatherMask, vGatherMask, vQ1);
                uni_vmovups(vQ1, vGatherMask);
            }
        }
        uni_vfmsub231ps(vQ1, vAux, vDX);  // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3);  // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }

        RegPrinter::print<float>(*this, vQ1.reg, "Res0 out");
        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            store(ptr[rDstTmp], vQ1, regWorkAmount, dataTypeSize);
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <>
void GridSample3DKernel<x64::avx512_core>::bilinearInterpolation2D1(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord, const Vmm& vDZ, bool tail) {
    // const auto& vDX = vWCoord;
    // const auto& vDY = vHCoord;
    auto vDX = getVmm();
    auto vDY = getVmm();
    auto shift00 = getVmm();
    auto shift01 = getVmm();
    auto shift10 = getVmm();
    auto shift11 = getVmm();
    auto vAux = getVmm();
    RegistersPool::Reg<Vmask> kMask00, kMask01, kMask10, kMask11;

    uni_vroundps(shift00, vWCoord, 0x1);  // Round floor x0
    uni_vroundps(shift10, vHCoord, 0x1);  // Round floor y0
    uni_vsubps(vDX, vWCoord, shift00);  
    uni_vsubps(vDY, vHCoord, shift01);
    uni_vaddps(shift01, shift00, vOnesF); // x1
    uni_vaddps(shift11, shift10, vOnesF); // y1  fisrt --x==0,y==1, second --number 0/1
    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        kMask00 = getMask();
        kMask01 = getMask();
        kMask10 = getMask();
        kMask11 = getMask();

        zerosPadding(kMask00, vDCoord, shift10, shift00);  // (z, y0; x0)
        zerosPadding(kMask01, vDCoord, shift10, shift01);  // (z, y0; x1)
        zerosPadding(kMask11, vDCoord, shift11, shift01);  // (z, y1; x1)
        zerosPadding(kMask10, vDCoord, shift11, shift00);  // (z, y1; x0)

        hwShiftPs2dq(shift00, shift10, shift00, vSrcWidthF);// y0,x0
        //TODO: add z offset z*vSrcWidthF*vSrcHeightF
        uni_vfmadd231ps(shift00, vDCoord, vSrcWidthHeightB);  //z, y0, x0, mask00

        uni_vpaddd(shift01, shift00, vDataTypeSizeB); // z, y0, x1, mask01
        uni_vpaddd(shift10, shift00, vSrcWidthB); //z, y1, x0, mask10
        uni_vpaddd(shift11, shift10, vDataTypeSizeB);//z, y1, x1, mask11
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(shift00, shift00, coord::w);
        borderPadding(shift01, shift01, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(shift00, shift00, coord::w);
        reflectionPadding(shift01, shift01, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (jcp.paddingMode == GridSamplePaddingMode::BORDER || jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, shift00, vSrcWidthF);
        hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
        hwShiftPs2dq(shift01, shift01, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto kAuxMask = getMask();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.channelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, 0);

            L(lChannelLoopBegin);
            cmp(rChannel, regChannelNum);
            jge(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill);  // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0);  // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
        }
        gatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX);  // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
        }
        gatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1);   // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux, vDX);  // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);
        // uni_vmulps(vQ1, vQ1, vDZ); // Res = Res * dz

        vmovaps(vQ0, ptr[rDstTmp]);
        uni_vfmadd132ps(vQ1, vQ0, vDZ); // Res = LastDxyz + Res * dz    

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3);  // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }

        RegPrinter::print<float>(*this, vQ1.reg, "Res1 out");
        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            inc(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::bilinearInterpolation2D1(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord,
    const Vmm& vDZ, bool tail) {
    auto vWRound = getVmm();
    auto vHRound = getVmm();
    auto& vDX = vWCoord;
    auto& vDY = vHCoord;
    auto vAux = getVmm();
    Vmm shift00, shift01, shift10, shift11;
    RegistersPool::Reg<Vmm> shift10Holder, shift11Holder;
    // For ZEROS padding only.
    RegistersPool::Reg<Vmm> vMask00, vMask01, vMask10, vMask11;

    uni_vroundps(vWRound, vWCoord, 0x1);  // Round floor
    uni_vroundps(vHRound, vHCoord, 0x1);  // Round floor
    uni_vsubps(vDX, vDX, vWRound);
    uni_vsubps(vDY, vDY, vHRound);

    if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
        shift00 = vWRound;
        shift01 = vHRound;
        shift10Holder = getVmm();
        shift10 = shift10Holder;
        shift11Holder = getVmm();
        shift11 = shift11Holder;

        uni_vaddps(shift10, vWRound, vOnesF);
        uni_vaddps(shift11, vHRound, vOnesF);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        {
            auto rAux = getReg64();
            static const float onesArr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
            if (isa == x64::sse41) {
                static const float* onesPtr = onesArr + (reinterpret_cast<int64_t>(onesArr) % 16) / sizeof(float);
                mov(rAux, reinterpret_cast<uintptr_t>(onesPtr));
            } else {
                mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
            }
            uni_vmovups(vAux, ptr[rAux]);
        }
        shift00 = vWRound;
        shift10 = vHRound;
        vMask00 = getVmm();
        vMask01 = getVmm();
        vMask10 = getVmm();
        vMask11 = getVmm();

        uni_vaddps(vMask00, vWRound, vAux);
        uni_vaddps(vAux, vAux, vHRound);

        zerosPadding(vMask01, vDCoord, vHRound, vMask00);  // (y; x + 1)
        zerosPadding(vMask10, vDCoord, vAux, vWRound);     // (y + 1; x)
        zerosPadding(vMask11, vDCoord, vAux, vMask00);     // (y + 1; x + 1)
        zerosPadding(vMask00, vDCoord, vHRound, vWRound);  // (y; x)

        hwShiftPs2dq(shift00, vHRound, vWRound, vSrcWidthF);
        //TODO: add z offset z*vSrcWidthF*vSrcHeightF
        uni_vfmadd231ps(shift00, vDCoord, vSrcWidthHeightB);  
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(vWRound, vWRound, coord::w);
        borderPadding(vHRound, vHRound, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(vWRound, vWRound, coord::w);
        reflectionPadding(vHRound, vHRound, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::REFLECTION)) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWRound, vSrcWidthF);
        hwShiftPs2dq(vWRound, vHRound, vWRound, vSrcWidthF);
        hwShiftPs2dq(vHRound, vHRound, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto vGatherMask = getVmm();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    auto rTypeSize = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    mov(rTypeSize, ptr[regParams + GET_OFF(dataTypeSize)]);

    for (uint64_t ch = 0; ch < jcp.channelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS && isa == x64::avx2) {
            uni_vmovups(vGatherMask, vMask00);
        }
        gatherdd(vQ0,
                 rSrcTmp,
                 shift00,
                 (isa == x64::avx2 || !vMask00.isInitialized()) ? vGatherMask : vMask00,
                 useMask,
                 zeroFill);  // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ0, vDX, vQ0);  // q0 = -(v00 - dx * v00)
        } else {
            uni_vmulps(vGatherMask, vQ0, vDX);
            uni_vsubps(vQ0, vQ0, vGatherMask);
        }

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpaddd(shift10, shift00, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask01);
        }
        gatherdd(vAux,
                 rSrcTmp,
                 jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift01 : shift10,
                 (isa == x64::avx2 || !vMask01.isInitialized()) ? vGatherMask : vMask01,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        if (isa == x64::avx2) {
            uni_vfmsub231ps(vQ0, vAux, vDX);  // q0 = -q0 + dx * v01
        } else {
            uni_vmulps(vAux, vAux, vDX);
            uni_vaddps(vQ0, vQ0, vAux);
        }

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            {
                auto rSrcWidth = getReg64();
                mov(rSrcWidth, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpaddd(shift10, shift10, ptr[rSrcWidth]);
            }
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask11);
        }
        gatherdd(vAux,
                 rSrcTmp,
                 jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift11 : shift10,
                 (isa == x64::avx2 || !vMask11.isInitialized()) ? vGatherMask : vMask11,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpsubd(shift10, shift10, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask10);
        }
        gatherdd(vQ1,
                 rSrcTmp,
                 shift10,
                 (isa == x64::avx2 || !vMask10.isInitialized()) ? vGatherMask : vMask10,
                 useMask,
                 zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        // q1 = -(v10 - dx * v10)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ1, vDX, vQ1);
        } else {
            uni_vmulps(vGatherMask, vQ1, vDX);
            if (isa == x64::avx) {
                uni_vsubps(vQ1, vGatherMask, vQ1);
            } else {
                uni_vsubps(vGatherMask, vGatherMask, vQ1);
                uni_vmovups(vQ1, vGatherMask);
            }
        }
        uni_vfmsub231ps(vQ1, vAux, vDX);  // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3);  // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }

        RegPrinter::print<float>(*this, vQ1.reg, "Res1 out");

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            store(ptr[rDstTmp], vQ1, regWorkAmount, dataTypeSize);
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, AVX, SSE41
void GridSample3DKernel<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vDCoord, bool tail) {
    auto z1 = getVmm();
    auto dz1 = getVmm();
    auto z2 = getVmm();
    auto dz2 = getVmm();
    // const Vmm& z2 = vDCoord;
    // const Vmm& dz2 = vDCoord;

    uni_vroundps(z1, vDCoord, 0x1);  // Round floor
    uni_vsubps(dz1, vDCoord, z1);
    uni_vaddps(z2, z1, vOnesF);
    bilinearInterpolation2D0(vWCoord, vHCoord, z2, dz1, tail);
    uni_vsubps(dz2, vOnesF, dz1);
    bilinearInterpolation2D1(vWCoord, vHCoord, z1, dz2, tail);
}

template <x64::cpu_isa_t isa>
void GridSample3DKernel<isa>::dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc) {
    if (dataTypeSize == 1)
        return;

    if (isa == x64::avx) {  // vpslld works just with XMM for AVX, so use vmulps for YMM
        auto rAux = getReg64();
        static const float val = dataTypeSize;
        static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
        mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
        uni_vmulps(vDst, vSrc, ptr[rAux]);
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vSrc);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift);  // multiply by source data type size.
    }
}

template <x64::cpu_isa_t isa>
void GridSample3DKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        if (one_of(isa, x64::avx512_core, x64::avx2)) {
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            auto vTmp = getVmm();
            uni_vmulps(vTmp, vHCoord, vWidth);
            uni_vaddps(vDst, vWCoord, vTmp);
        }
    } else if (vDst.getIdx() == vHCoord.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vWidth);
    } else if (vDst.getIdx() == vWidth.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vHCoord);
    } else {
        if (one_of(isa, x64::avx2, x64::avx512_core)) {
            uni_vmovups(vDst, vWCoord);
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            uni_vmulps(vDst, vHCoord, vWidth);
            uni_vaddps(vDst, vDst, vWCoord);
        }
    }

    if (isa == x64::avx) {  // vpslld works just with XMM for AVX, so use vmulps for YMM
        if (dataTypeSize > 1) {
            auto rAux = getReg64();
            const float val = dataTypeSize;
            static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
            mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[rAux]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vDst);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift);  // multiply by source data type size.
    }
}

template class GridSample3DKernel<x64::avx512_core>;
template class GridSample3DKernel<x64::avx2>;
template class GridSample3DKernel<x64::sse41>;

}  // namespace kernel
}  // namespace intel_cpu
}  // namespace ov
