#pragma once

#include <string>
#include <iostream>
#include <cstdint>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>
#endif

namespace matrix {

struct CpuFeatures {
    bool sse = false;
    bool sse2 = false;
    bool sse3 = false;
    bool ssse3 = false;
    bool sse4_1 = false;
    bool sse4_2 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    std::string brand_string;
    uint32_t num_logical_cores = 0;

    static const CpuFeatures& get() {
        static CpuFeatures instance = detect();
        return instance;
    }

    void print(std::ostream& os) const {
        os << "CPU: " << (brand_string.empty() ? "Unknown x86/x64" : brand_string) << "\n";
        os << "Features: "
           << "AVX=" << (avx ? "yes" : "no") << ", "
           << "AVX2=" << (avx2 ? "yes" : "no") << ", "
           << "FMA=" << (fma ? "yes" : "no") << ", "
           << "SSE4.1=" << (sse4_1 ? "yes" : "no") << ", "
           << "SSE4.2=" << (sse4_2 ? "yes" : "no") << ", "
           << "AVX512F=" << (avx512f ? "yes" : "no") << "\n";
    }

private:
    static void run_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
#if defined(_MSC_VER)
        __cpuidex(reinterpret_cast<int*>(regs), static_cast<int>(leaf), static_cast<int>(subleaf));
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
        regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
    }

    static CpuFeatures detect() {
        CpuFeatures feat;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        uint32_t regs[4] = {0};

        // Query maximum standard leaf
        run_cpuid(0, 0, regs);
        uint32_t max_leaf = regs[0];

        if (max_leaf >= 1) {
            run_cpuid(1, 0, regs);
            uint32_t ecx = regs[2];
            uint32_t edx = regs[3];

            feat.sse    = (edx & (1u << 25)) != 0;
            feat.sse2   = (edx & (1u << 26)) != 0;
            feat.sse3   = (ecx & (1u << 0))  != 0;
            feat.ssse3  = (ecx & (1u << 9))  != 0;
            feat.fma    = (ecx & (1u << 12)) != 0;
            feat.sse4_1 = (ecx & (1u << 19)) != 0;
            feat.sse4_2 = (ecx & (1u << 20)) != 0;

            bool osxsave = (ecx & (1u << 27)) != 0;
            bool cpu_avx = (ecx & (1u << 28)) != 0;

            // Check if OS supports AVX context saving (XGETBV)
            if (osxsave && cpu_avx) {
#if defined(_MSC_VER)
                unsigned __int64 xcr0 = _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
                uint32_t xcr0_low, xcr0_high;
                __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
                uint64_t xcr0 = (static_cast<uint64_t>(xcr0_high) << 32) | xcr0_low;
#else
                uint64_t xcr0 = 0;
#endif
                // XCR0 bits 1 and 2 indicate XMM and YMM state support
                if ((xcr0 & 0x6) == 0x6) {
                    feat.avx = true;
                }
            }
        }

        if (max_leaf >= 7 && feat.avx) {
            run_cpuid(7, 0, regs);
            uint32_t ebx = regs[1];
            feat.avx2    = (ebx & (1u << 5)) != 0;
            feat.avx512f = (ebx & (1u << 16)) != 0;
        }

        // Query brand string (leaves 0x80000002 to 0x80000004)
        run_cpuid(0x80000000, 0, regs);
        uint32_t max_ext_leaf = regs[0];
        if (max_ext_leaf >= 0x80000004) {
            char brand[49] = {0};
            for (uint32_t i = 0; i < 3; ++i) {
                run_cpuid(0x80000002 + i, 0, regs);
                reinterpret_cast<uint32_t*>(brand)[i * 4 + 0] = regs[0];
                reinterpret_cast<uint32_t*>(brand)[i * 4 + 1] = regs[1];
                reinterpret_cast<uint32_t*>(brand)[i * 4 + 2] = regs[2];
                reinterpret_cast<uint32_t*>(brand)[i * 4 + 3] = regs[3];
            }
            feat.brand_string = brand;
            // Trim leading/trailing whitespace
            size_t start = feat.brand_string.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                size_t end = feat.brand_string.find_last_not_of(" \t\n\r");
                feat.brand_string = feat.brand_string.substr(start, end - start + 1);
            }
        }
#endif
        return feat;
    }
};

} // namespace matrix
