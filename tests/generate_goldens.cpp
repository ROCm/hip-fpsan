// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Extensible silicon golden generator. CDNA3 matrix intrinsics are the first
// registered family; future architectures should add registry headers and
// adapter emitters rather than bespoke generators.
//
// Flow:
//   1. Build a few deterministic logical A/B/C/(idx) matrices.
//   2. In a one-wave HIP kernel, pack the current lane's fragment exactly as the
//      target intrinsic expects.
//   3. Call the raw clang __builtin_amdgcn_* intrinsic.
//   4. Decode the lane-local D registers back into logical matrix order.
//   5. Emit all inputs and outputs as a C++ header consumed by the tests.
//
// This intentionally makes the checked-in outputs an oracle for wrapper
// forwarding. The logical-matrix-to-fragment packing still uses the same layout
// knowledge as the tests, so this is not an independent proof of those helpers.
#include "fpsan/amdgcn_mfma.hpp"
#include "fpsan/amdgcn_smfmac.hpp"
#include "fpsan/fpsan.hpp"

#include "golden_cdna3_matrix_registry.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC  = Conversions::Explicit;
static constexpr int         WAVE = 64;

// Centralized golden input policy. Each case gives explicit value streams for
// A/B/C and an explicit sparse-pair table. Matrix builders consume the streams
// cyclically, so a compact sequence can describe large matrices while still
// making the actual fixture values visible and easy to replace.
struct InputSequence
{
    const float* values;
    std::size_t  count;
};

template <std::size_t N>
constexpr InputSequence input_sequence(const float (&values)[N])
{
    return {values, N};
}

struct SparsePairChoice
{
    int p0;
    int p1;
};

struct SparsePairPattern
{
    const SparsePairChoice* pairs;
    std::size_t             count;
    int                     row_period;
    int                     group_period;
};

template <std::size_t N>
constexpr SparsePairPattern
    sparse_pattern(const SparsePairChoice (&pairs)[N], int row_period, int group_period)
{
    return {pairs, N, row_period, group_period};
}

struct GoldenInputCase
{
    InputSequence     a;
    InputSequence     b;
    InputSequence     c;
    SparsePairPattern sparse;
};

// Each input case is declared once here. The tuple is:
//   X(Name, sparse_row_period, sparse_group_period, A values, B values, C values, sparse pairs)
// A/B/C sequences are repeated cyclically to fill each logical matrix. Sparse
// pairs are indexed as row_period x group_period and repeated over rows/groups.
#define FPSAN_GOLDEN_INPUT_CASES(X)                                     \
    X(Case0,                                                            \
      4,                                                                \
      2,                                                                \
      (1.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f),                                                           \
      (1.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f,                                                            \
       0.0f),                                                           \
      (-2.0f, -1.0f, 0.0f, 1.0f, 2.0f),                                 \
      ({0, 1}, {0, 2}, {1, 3}, {0, 3}, {2, 3}, {0, 2}, {1, 3}, {1, 2})) \
    X(Case1,                                                            \
      4,                                                                \
      2,                                                                \
      (2.0f, -3.0f, -1.0f, 1.0f, 3.0f, -2.0f, 0.0f),                    \
      (-1.0f, 1.0f, 3.0f, -2.0f, 0.0f, 2.0f, -3.0f),                    \
      (-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f),       \
      ({1, 3}, {0, 3}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {0, 1}, {0, 2})) \
    X(Case2,                                                            \
      4,                                                                \
      2,                                                                \
      (-1.0f, 1.0f),                                                    \
      (0.25f, -4.0f),                                                   \
      (2.0f, -0.5f),                                                    \
      ({2, 3}, {0, 2}, {1, 3}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {0, 3}))

#define FPSAN_UNPAREN(...) __VA_ARGS__

#define FPSAN_DECLARE_INPUT_CASE(                                                       \
    Name, RowPeriod, GroupPeriod, AValues, BValues, CValues, SparseValues)              \
    static constexpr float            k##Name##AInputs[]     = {FPSAN_UNPAREN AValues}; \
    static constexpr float            k##Name##BInputs[]     = {FPSAN_UNPAREN BValues}; \
    static constexpr float            k##Name##CInputs[]     = {FPSAN_UNPAREN CValues}; \
    static constexpr SparsePairChoice k##Name##SparsePairs[] = {FPSAN_UNPAREN SparseValues};

FPSAN_GOLDEN_INPUT_CASES(FPSAN_DECLARE_INPUT_CASE)

#undef FPSAN_DECLARE_INPUT_CASE

#define FPSAN_REGISTER_INPUT_CASE(                                         \
    Name, RowPeriod, GroupPeriod, AValues, BValues, CValues, SparseValues) \
    {input_sequence(k##Name##AInputs),                                     \
     input_sequence(k##Name##BInputs),                                     \
     input_sequence(k##Name##CInputs),                                     \
     sparse_pattern(k##Name##SparsePairs, RowPeriod, GroupPeriod)},

static constexpr GoldenInputCase kGoldenInputCases[]
    = {FPSAN_GOLDEN_INPUT_CASES(FPSAN_REGISTER_INPUT_CASE)};

#undef FPSAN_REGISTER_INPUT_CASE
#undef FPSAN_UNPAREN

static constexpr int kGoldenCaseCount
    = static_cast<int>(sizeof(kGoldenInputCases) / sizeof(kGoldenInputCases[0]));

// Most raw MFMA builtins accept the same native vector type used by the fpsan
// wrapper ABI. The fp8/bf8 MFMA builtins are the exception: clang exposes their
// per-lane 8-byte fragment as a `long`, so adapt those fragments here.
template <class T>
__device__ T raw_mfma_arg(T v)
{
    return v;
}

template <class Elem>
__device__ long raw_mfma_arg(fpsan::detail::v8_fragment<Elem> v)
{
    return __builtin_bit_cast(long, v);
}

#define HIP_CHECK_TOOL(e)                                                \
    do                                                                   \
    {                                                                    \
        hipError_t e_ = (e);                                             \
        if(e_ != hipSuccess)                                             \
        {                                                                \
            std::cerr << "HIP error: " << hipGetErrorString(e_) << "\n"; \
            std::exit(1);                                                \
        }                                                                \
    } while(0)

template <class T>
std::uint64_t bits_of_value(T v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof v);
    return bits;
}

template <class T>
std::vector<std::uint64_t> bits_of_vector(const std::vector<T>& values)
{
    std::vector<std::uint64_t> bits(values.size());
    for(std::size_t i = 0; i < values.size(); ++i)
        bits[i] = bits_of_value(values[i]);
    return bits;
}

template <class T>
T* to_dev_tool(const std::vector<T>& host)
{
    T* ptr = nullptr;
    HIP_CHECK_TOOL(hipMalloc(&ptr, host.size() * sizeof(T)));
    HIP_CHECK_TOOL(hipMemcpy(ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return ptr;
}

template <class T>
std::vector<T> from_dev_tool(const T* ptr, std::size_t count)
{
    std::vector<T> host(count);
    HIP_CHECK_TOOL(hipMemcpy(host.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
}

const GoldenInputCase& golden_input_case(int case_id)
{
    if(case_id < 0 || case_id >= kGoldenCaseCount)
    {
        std::cerr << "Invalid golden input case: " << case_id << "\n";
        std::exit(1);
    }
    return kGoldenInputCases[case_id];
}

float sequence_value(const InputSequence& sequence, int index)
{
    if(sequence.count == 0)
    {
        std::cerr << "Golden input sequence must not be empty\n";
        std::exit(1);
    }
    return sequence.values[static_cast<std::size_t>(index) % sequence.count];
}

template <class T>
T typed_sequence_value(const InputSequence& sequence, int index)
{
    return static_cast<T>(sequence_value(sequence, index));
}

SparsePairChoice sparse_pair(const SparsePairPattern& sparse, int row, int group)
{
    if(sparse.row_period <= 0 || sparse.group_period <= 0)
    {
        std::cerr << "Sparse pair pattern periods must be positive\n";
        std::exit(1);
    }
    const std::size_t index
        = static_cast<std::size_t>(row % sparse.row_period) * sparse.group_period
          + static_cast<std::size_t>(group % sparse.group_period);
    if(index >= sparse.count)
    {
        std::cerr << "Sparse pair pattern is too small for configured periods\n";
        std::exit(1);
    }
    return sparse.pairs[index];
}

// The generated header stores everything as uint64_t bit patterns. Tests cast
// those bits back to the required element type, which keeps fp8/bf8/f16/bf16/f32
// and f64 cases in one simple fixture format.
std::string array_name(const char* name, int case_id, const char* field)
{
    std::ostringstream os;
    os << "k" << name << "_Case" << case_id << "_" << field;
    return os.str();
}

void emit_array(const std::string& name, const std::vector<std::uint64_t>& values)
{
    std::cout << "    inline constexpr std::uint64_t " << name << "[] = {";
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        if(i % 8 == 0)
            std::cout << "\n        ";
        else
            std::cout << ' ';
        std::cout << "0x" << std::hex << values[i] << std::dec << "ull";
        if(i + 1 != values.size())
            std::cout << ",";
    }
    if(!values.empty())
        std::cout << "\n    ";
    std::cout << "};\n\n";
}

struct EmittedCase
{
    std::string name;
    int         case_id;
    std::string a;
    std::size_t a_count;
    std::string b;
    std::size_t b_count;
    std::string c;
    std::size_t c_count;
    std::string idx;
    std::size_t idx_count;
    std::string d;
    std::size_t d_count;
};

std::vector<EmittedCase> g_cases;

// Emit the raw arrays immediately, then remember enough metadata to emit the
// CaseView table at the end. This keeps generator memory use small even though
// the checked-in header is large.
void record_case(const char*                       name,
                 int                               case_id,
                 const std::vector<std::uint64_t>& a,
                 const std::vector<std::uint64_t>& b,
                 const std::vector<std::uint64_t>& c,
                 const std::vector<std::uint64_t>& idx,
                 const std::vector<std::uint64_t>& d)
{
    const std::string an = array_name(name, case_id, "A");
    const std::string bn = array_name(name, case_id, "B");
    const std::string cn = array_name(name, case_id, "C");
    const std::string dn = array_name(name, case_id, "D");
    emit_array(an, a);
    emit_array(bn, b);
    emit_array(cn, c);
    std::string in;
    if(!idx.empty())
    {
        in = array_name(name, case_id, "Idx");
        emit_array(in, idx);
    }
    emit_array(dn, d);
    g_cases.push_back(
        {name, case_id, an, a.size(), bn, b.size(), cn, c.size(), in, idx.size(), dn, d.size()});
}

// ---------------------------------------------------------------------------
// Dense vector-input MFMA.
// ---------------------------------------------------------------------------
template <class Traits>
struct DenseVecHarness
{
    using AVec             = typename Traits::AVec;
    using BVec             = typename Traits::BVec;
    using CVec             = typename Traits::CVec;
    using AElem            = fpsan::detail::vector_element_t<AVec>;
    using BElem            = fpsan::detail::vector_element_t<BVec>;
    using CElem            = fpsan::detail::vector_element_t<CVec>;
    static constexpr int M = Traits::M, N = Traits::N, K = Traits::K, Bk = Traits::Bk;
    static constexpr int InBits    = Traits::InBits;
    static constexpr int per_dword = 32 / InBits;
};

template <class Traits>
struct DenseVecData
{
    using H = DenseVecHarness<Traits>;
    std::vector<typename H::AElem> A;
    std::vector<typename H::BElem> B;
    std::vector<typename H::CElem> C;
};

template <class Traits>
DenseVecData<Traits> make_dense_vec_case(int case_id)
{
    using H                     = DenseVecHarness<Traits>;
    const auto&          inputs = golden_input_case(case_id);
    DenseVecData<Traits> d;
    d.A.resize(H::Bk * H::M * H::K);
    d.B.resize(H::Bk * H::K * H::N);
    d.C.resize(H::Bk * H::M * H::N);
    for(std::size_t i = 0; i < d.A.size(); ++i)
        d.A[i] = typed_sequence_value<typename H::AElem>(inputs.a, static_cast<int>(i));
    for(std::size_t i = 0; i < d.B.size(); ++i)
        d.B[i] = typed_sequence_value<typename H::BElem>(inputs.b, static_cast<int>(i));
    for(std::size_t i = 0; i < d.C.size(); ++i)
        d.C[i] = typed_sequence_value<typename H::CElem>(inputs.c, static_cast<int>(i));
    return d;
}

template <class Traits, Semantics S>
__device__ void load_dense_vec(const typename DenseVecHarness<Traits>::AElem*         A,
                               const typename DenseVecHarness<Traits>::BElem*         B,
                               const typename DenseVecHarness<Traits>::CElem*         C,
                               int                                                    lane,
                               Value<typename DenseVecHarness<Traits>::AVec, S, kCC>& a,
                               Value<typename DenseVecHarness<Traits>::BVec, S, kCC>& b,
                               Value<typename DenseVecHarness<Traits>::CVec, S, kCC>& c)
{
    using H = DenseVecHarness<Traits>;
    typename H::AVec an{};
    typename H::BVec bn{};
    typename H::CVec cn{};
    // Convert from logical matrix order into the per-lane A/B/C register
    // fragments consumed by the raw builtin. The helper returns which lane,
    // dword register, and sub-dword element owns each logical input.
    for(int blk = 0; blk < H::Bk; ++blk)
        for(int i = 0; i < H::M; ++i)
            for(int k = 0; k < H::K; ++k)
            {
                auto loc = fpsan::detail::input_loc(H::M, H::K, H::Bk, i, k, blk, H::InBits);
                if(loc.lane == lane)
                    an[H::per_dword * loc.reg + loc.sub] = A[(blk * H::M + i) * H::K + k];
            }
    for(int blk = 0; blk < H::Bk; ++blk)
        for(int j = 0; j < H::N; ++j)
            for(int k = 0; k < H::K; ++k)
            {
                auto loc = fpsan::detail::input_loc(H::N, H::K, H::Bk, j, k, blk, H::InBits);
                if(loc.lane == lane)
                    bn[H::per_dword * loc.reg + loc.sub] = B[(blk * H::K + k) * H::N + j];
            }
    for(int blk = 0; blk < H::Bk; ++blk)
        for(int i = 0; i < H::M; ++i)
            for(int j = 0; j < H::N; ++j)
            {
                auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, blk);
                if(loc.lane == lane)
                    cn[loc.reg] = C[(blk * H::M + i) * H::N + j];
            }
    a = Value<typename H::AVec, S, kCC>(an);
    b = Value<typename H::BVec, S, kCC>(bn);
    c = Value<typename H::CVec, S, kCC>(cn);
}

template <class Traits>
__global__ void k_dense_vec_float(const typename DenseVecHarness<Traits>::AElem* A,
                                  const typename DenseVecHarness<Traits>::BElem* B,
                                  const typename DenseVecHarness<Traits>::CElem* C,
                                  typename DenseVecHarness<Traits>::CElem*       D)
{
    using H                                             = DenseVecHarness<Traits>;
    int                                            lane = threadIdx.x;
    Value<typename H::AVec, Semantics::Float, kCC> a;
    Value<typename H::BVec, Semantics::Float, kCC> b;
    Value<typename H::CVec, Semantics::Float, kCC> c;
    load_dense_vec<Traits, Semantics::Float>(A, B, C, lane, a, b, c);
    // Capture through the raw clang builtin. Tests later run the fpsan wrapper
    // on these same fragments and compare to this output.
    auto d = Traits::call_raw(a.to_float(), b.to_float(), c.to_float());
    // Decode the per-lane accumulator registers back into row-major logical D.
    for(int blk = 0; blk < H::Bk; ++blk)
        for(int i = 0; i < H::M; ++i)
            for(int j = 0; j < H::N; ++j)
            {
                auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, blk);
                if(loc.lane == lane)
                    D[(blk * H::M + i) * H::N + j] = d[loc.reg];
            }
}

template <class Traits>
void emit_dense_vec(const char* name)
{
    using H = DenseVecHarness<Traits>;
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto               d  = make_dense_vec_case<Traits>(case_id);
        auto*              dA = to_dev_tool(d.A);
        auto*              dB = to_dev_tool(d.B);
        auto*              dC = to_dev_tool(d.C);
        typename H::CElem* dD = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, d.C.size() * sizeof(typename H::CElem)));
        k_dense_vec_float<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, d.C.size());
        record_case(name,
                    case_id,
                    bits_of_vector(d.A),
                    bits_of_vector(d.B),
                    bits_of_vector(d.C),
                    {},
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }
}

#define DEFINE_DENSE_VEC_TRAITS(Name, M_, N_, K_, Bk_, InBits_, AV_, BV_, CV_, WRAP_) \
    struct Name                                                                       \
    {                                                                                 \
        using AVec               = AV_;                                               \
        using BVec               = BV_;                                               \
        using CVec               = CV_;                                               \
        static constexpr int   M = M_, N = N_, K = K_, Bk = Bk_, InBits = InBits_;    \
        __device__ static CVec call_raw(AVec a, BVec b, CVec c)                       \
        {                                                                             \
            /* WRAP_ names match the builtin suffix, so this expands to the raw */    \
            /* clang __builtin_amdgcn_* call rather than the fpsan:: wrapper. */      \
            return __builtin_##WRAP_(raw_mfma_arg(a), raw_mfma_arg(b), c, 0, 0, 0);   \
        }                                                                             \
    };

using fpsan::v16f_native;
using fpsan::v2f_native;
using fpsan::v32f_native;
using fpsan::v4bf_native;
using fpsan::v4f_native;
using fpsan::v4h_native;
using fpsan::v8amd_e4m3_native;
using fpsan::v8amd_e5m2_native;

FPSAN_CDNA3_DENSE_VEC_MATRIX_INTRINSICS(DEFINE_DENSE_VEC_TRAITS)

#undef DEFINE_DENSE_VEC_TRAITS

// ---------------------------------------------------------------------------
// Dense scalar f32-input MFMA.
// ---------------------------------------------------------------------------
template <class Traits>
struct DenseF32Data
{
    std::vector<float> A, B, C;
};

template <class Traits>
DenseF32Data<Traits> make_dense_f32_case(int case_id)
{
    const auto&          inputs = golden_input_case(case_id);
    DenseF32Data<Traits> d;
    d.A.resize(Traits::Bk * Traits::M * Traits::K);
    d.B.resize(Traits::Bk * Traits::K * Traits::N);
    d.C.resize(Traits::Bk * Traits::M * Traits::N);
    for(std::size_t i = 0; i < d.A.size(); ++i)
        d.A[i] = typed_sequence_value<float>(inputs.a, static_cast<int>(i));
    for(std::size_t i = 0; i < d.B.size(); ++i)
        d.B[i] = typed_sequence_value<float>(inputs.b, static_cast<int>(i));
    for(std::size_t i = 0; i < d.C.size(); ++i)
        d.C[i] = typed_sequence_value<float>(inputs.c, static_cast<int>(i));
    return d;
}

template <class Traits>
__global__ void k_dense_f32_float(const float* A, const float* B, const float* C, float* D)
{
    using T    = Traits;
    int   lane = threadIdx.x;
    float an = 0.0f, bn = 0.0f;
    for(int blk = 0; blk < T::Bk; ++blk)
        for(int i = 0; i < T::M; ++i)
            for(int k = 0; k < T::K; ++k)
            {
                auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, blk, 32);
                if(loc.lane == lane)
                    an = A[(blk * T::M + i) * T::K + k];
            }
    for(int blk = 0; blk < T::Bk; ++blk)
        for(int j = 0; j < T::N; ++j)
            for(int k = 0; k < T::K; ++k)
            {
                auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, blk, 32);
                if(loc.lane == lane)
                    bn = B[(blk * T::K + k) * T::N + j];
            }
    typename T::CVec cn{};
    for(int blk = 0; blk < T::Bk; ++blk)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto loc = fpsan::detail::output_loc_32(T::M, T::N, i, j, blk);
                if(loc.lane == lane)
                    cn[loc.reg] = C[(blk * T::M + i) * T::N + j];
            }
    Value<float, Semantics::Float, kCC>            a{an}, b{bn};
    Value<typename T::CVec, Semantics::Float, kCC> c{cn};
    // Scalar-input MFMA has the same logical flow as dense vector MFMA, but A and
    // B are one f32 per lane rather than vector fragments.
    auto d = T::call_raw(a.to_float(), b.to_float(), c.to_float());
    for(int blk = 0; blk < T::Bk; ++blk)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto loc = fpsan::detail::output_loc_32(T::M, T::N, i, j, blk);
                if(loc.lane == lane)
                    D[(blk * T::M + i) * T::N + j] = d[loc.reg];
            }
}

template <class Traits>
void emit_dense_f32(const char* name)
{
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto   d  = make_dense_f32_case<Traits>(case_id);
        auto*  dA = to_dev_tool(d.A);
        auto*  dB = to_dev_tool(d.B);
        auto*  dC = to_dev_tool(d.C);
        float* dD = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, d.C.size() * sizeof(float)));
        k_dense_f32_float<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, d.C.size());
        record_case(name,
                    case_id,
                    bits_of_vector(d.A),
                    bits_of_vector(d.B),
                    bits_of_vector(d.C),
                    {},
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }
}

#define DEFINE_DENSE_F32_TRAITS(Name, M_, N_, K_, Bk_, CVec_, WRAP_) \
    struct Name                                                      \
    {                                                                \
        using CVec               = CVec_;                            \
        static constexpr int   M = M_, N = N_, K = K_, Bk = Bk_;     \
        __device__ static CVec call_raw(float a, float b, CVec c)    \
        {                                                            \
            return __builtin_##WRAP_(a, b, c, 0, 0, 0);              \
        }                                                            \
    };

FPSAN_CDNA3_DENSE_F32_MATRIX_INTRINSICS(DEFINE_DENSE_F32_TRAITS)

#undef DEFINE_DENSE_F32_TRAITS

// ---------------------------------------------------------------------------
// F64 MFMA.
// ---------------------------------------------------------------------------
using fpsan::v4d_native;

static constexpr int F64_M = 16, F64_N = 16, F64_K = 4;
static constexpr int F64S_M = 4, F64S_N = 4, F64S_K = 4, F64S_B = 4;

struct F64Data
{
    std::vector<double> A, B, C;
};

F64Data make_f64_16_case(int case_id)
{
    const auto& inputs = golden_input_case(case_id);
    F64Data     d;
    d.A.resize(F64_M * F64_K);
    d.B.resize(F64_K * F64_N);
    d.C.resize(F64_M * F64_N);
    for(std::size_t i = 0; i < d.A.size(); ++i)
        d.A[i] = typed_sequence_value<double>(inputs.a, static_cast<int>(i));
    for(std::size_t i = 0; i < d.B.size(); ++i)
        d.B[i] = typed_sequence_value<double>(inputs.b, static_cast<int>(i));
    for(std::size_t i = 0; i < d.C.size(); ++i)
        d.C[i] = typed_sequence_value<double>(inputs.c, static_cast<int>(i));
    return d;
}

F64Data make_f64_4_case(int case_id)
{
    const auto& inputs = golden_input_case(case_id);
    F64Data     d;
    d.A.resize(F64S_B * F64S_M * F64S_K);
    d.B.resize(F64S_B * F64S_K * F64S_N);
    d.C.resize(F64S_B * F64S_M * F64S_N);
    for(std::size_t i = 0; i < d.A.size(); ++i)
        d.A[i] = typed_sequence_value<double>(inputs.a, static_cast<int>(i));
    for(std::size_t i = 0; i < d.B.size(); ++i)
        d.B[i] = typed_sequence_value<double>(inputs.b, static_cast<int>(i));
    for(std::size_t i = 0; i < d.C.size(); ++i)
        d.C[i] = typed_sequence_value<double>(inputs.c, static_cast<int>(i));
    return d;
}

__global__ void k_f64_16x16x4_float(const double* A, const double* B, const double* C, double* D)
{
    int    lane = threadIdx.x;
    double an = 0.0, bn = 0.0;
    for(int i = 0; i < F64_M; ++i)
        for(int k = 0; k < F64_K; ++k)
        {
            auto loc = fpsan::detail::input_loc(F64_M, F64_K, 1, i, k, 0, 64);
            if(loc.lane == lane)
                an = A[i * F64_K + k];
        }
    for(int j = 0; j < F64_N; ++j)
        for(int k = 0; k < F64_K; ++k)
        {
            auto loc = fpsan::detail::input_loc(F64_N, F64_K, 1, j, k, 0, 64);
            if(loc.lane == lane)
                bn = B[k * F64_N + j];
        }
    v4d_native cn{};
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg / 2] = C[i * F64_N + j];
        }
    // The f64 16x16 shape follows the generic f64 input/output helper layout.
    auto d = __builtin_amdgcn_mfma_f64_16x16x4f64(an, bn, cn, 0, 0, 0);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
            if(loc.lane == lane)
                D[i * F64_N + j] = d[loc.reg / 2];
        }
}

__global__ void k_f64_4x4x4_float(const double* A, const double* B, const double* C, double* D)
{
    // f64 4x4x4 does not follow the dense helper formulas: one wave computes
    // four independent 4x4 blocks, with each lane holding one scalar output.
    const int    lane  = threadIdx.x;
    const int    in_k  = lane / 16;
    const int    blk   = (lane % 16) / 4;
    const int    idx   = lane % 4;
    const int    out_i = lane / 16;
    const int    out_j = lane % 4;
    const double a     = A[(blk * F64S_M + idx) * F64S_K + in_k];
    const double b     = B[(blk * F64S_K + in_k) * F64S_N + idx];
    const double c     = C[(blk * F64S_M + out_i) * F64S_N + out_j];
    auto         d     = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
    D[(blk * F64S_M + out_i) * F64S_N + out_j] = d;
}

void emit_f64_16()
{
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto    d  = make_f64_16_case(case_id);
        auto*   dA = to_dev_tool(d.A);
        auto*   dB = to_dev_tool(d.B);
        auto*   dC = to_dev_tool(d.C);
        double* dD = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, d.C.size() * sizeof(double)));
        k_f64_16x16x4_float<<<1, WAVE>>>(dA, dB, dC, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, d.C.size());
        record_case("MfmaF64_16x16x4",
                    case_id,
                    bits_of_vector(d.A),
                    bits_of_vector(d.B),
                    bits_of_vector(d.C),
                    {},
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }
}

void emit_f64_4()
{
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto    d  = make_f64_4_case(case_id);
        auto*   dA = to_dev_tool(d.A);
        auto*   dB = to_dev_tool(d.B);
        auto*   dC = to_dev_tool(d.C);
        double* dD = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, d.C.size() * sizeof(double)));
        k_f64_4x4x4_float<<<1, WAVE>>>(dA, dB, dC, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, d.C.size());
        record_case("MfmaF64_4x4x4",
                    case_id,
                    bits_of_vector(d.A),
                    bits_of_vector(d.B),
                    bits_of_vector(d.C),
                    {},
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }
}

// ---------------------------------------------------------------------------
// CDNA3 SMFMAC.
// ---------------------------------------------------------------------------
struct SparseData
{
    std::vector<float> A, B, C;
    std::vector<int>   idxbuf, p0, p1;
};

SparseData make_sparse_data(int case_id, int m, int n, int k, bool fp8_layout)
{
    const auto& inputs = golden_input_case(case_id);
    const int   groups = k / 4;
    const int   ccols  = 2 * groups;
    SparseData  d;
    d.A.resize(m * ccols);
    d.B.resize(k * n);
    d.C.resize(m * n);
    d.idxbuf.assign(WAVE, 0);
    d.p0.resize(m * groups);
    d.p1.resize(m * groups);
    for(std::size_t i = 0; i < d.A.size(); ++i)
        d.A[i] = sequence_value(inputs.a, static_cast<int>(i));
    for(std::size_t i = 0; i < d.B.size(); ++i)
        d.B[i] = sequence_value(inputs.b, static_cast<int>(i));
    for(std::size_t i = 0; i < d.C.size(); ++i)
        d.C[i] = sequence_value(inputs.c, static_cast<int>(i));
    for(int i = 0; i < m; ++i)
        for(int q = 0; q < groups; ++q)
        {
            auto [a0, a1]        = sparse_pair(inputs.sparse, i, q);
            d.p0[i * groups + q] = a0;
            d.p1[i * groups + q] = a1;
        }
    // SMFMAC takes a per-lane 16-bit sparse index. Each nibble describes one
    // 2:4 group: low two bits select the first live K offset, high two bits
    // select the second. The older f16/bf16 and fp8 instructions assign those
    // nibbles to lanes differently, hence the `fp8_layout` branch.
    if(fp8_layout)
    {
        const int half = groups / 2;
        for(int i = 0; i < m; ++i)
            for(int q = 0; q < groups; ++q)
            {
                const int lane = ((q % half) / 2) * m + i;
                const int nib  = 2 * (q / half) + (q % 2);
                d.idxbuf[lane] |= (d.p0[i * groups + q] | (d.p1[i * groups + q] << 2)) << (4 * nib);
            }
    }
    else
    {
        for(int i = 0; i < m; ++i)
            for(int q = 0; q < groups; ++q)
            {
                const int lane = (q / 2) * m + i;
                d.idxbuf[lane] |= (d.p0[i * groups + q] | (d.p1[i * groups + q] << 2))
                                  << (4 * (q % 2));
            }
    }
    return d;
}

template <class E, Semantics S, class Out>
__global__ void
    k_smf_16x16x32(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using v4e       = E __attribute__((ext_vector_type(4)));
    using v8e       = E __attribute__((ext_vector_type(8)));
    constexpr int N = 16, K = 32, Cc = K / 2;
    const int     lane = threadIdx.x;
    const int     g    = lane / 16;
    const int     j    = lane % 16;
    v4e           an{};
    for(int h = 0; h < 4; ++h)
        an[h] = static_cast<E>(A[j * Cc + g * 4 + h]);
    v8e bn{};
    for(int e = 0; e < 8; ++e)
        bn[e] = static_cast<E>(B[(g * 8 + e) * N + j]);
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * N + j];
    // CDNA3 f16/bf16 SMFMAC uses native vector operands directly in the builtin
    // ABI, so no packed integer bit-cast is needed here.
    auto d = [&] {
        if constexpr(std::is_same_v<E, _Float16>)
            return __builtin_amdgcn_smfmac_f32_16x16x32_f16(an, bn, cn, idx[lane], 0, 0);
        else
            return __builtin_amdgcn_smfmac_f32_16x16x32_bf16(an, bn, cn, idx[lane], 0, 0);
    }();
    for(int reg = 0; reg < 4; ++reg)
        D[(4 * g + reg) * N + j] = d[reg];
}

template <class E, Semantics S, class Out>
__global__ void
    k_smf_32x32x16(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using v4e       = E __attribute__((ext_vector_type(4)));
    using v8e       = E __attribute__((ext_vector_type(8)));
    constexpr int M = 32, N = 32, K = 16, Cc = K / 2;
    const int     lane = threadIdx.x;
    v4e           an{};
    for(int h = 0; h < 4; ++h)
        an[h] = static_cast<E>(A[(lane % 32) * Cc + (lane / 32) * 4 + h]);
    const int j    = (lane % 16) + 16 * ((lane / 16) % 2);
    const int kgrp = (lane / 16) / 2;
    v8e       bn{};
    for(int e = 0; e < 8; ++e)
        bn[e] = static_cast<E>(B[(8 * kgrp + e) * N + j]);
    fpsan::v16f_native cn{};
    for(int i = 0; i < M; ++i)
        for(int jj = 0; jj < N; ++jj)
        {
            auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * N + jj];
        }
    auto d = [&] {
        if constexpr(std::is_same_v<E, _Float16>)
            return __builtin_amdgcn_smfmac_f32_32x32x16_f16(an, bn, cn, idx[lane], 0, 0);
        else
            return __builtin_amdgcn_smfmac_f32_32x32x16_bf16(an, bn, cn, idx[lane], 0, 0);
    }();
    for(int i = 0; i < M; ++i)
        for(int jj = 0; jj < N; ++jj)
        {
            auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
            if(loc.lane == lane)
                D[i * N + jj] = d[loc.reg];
        }
}

template <class E>
void emit_smf_h(const char* name, int m, int n, int k)
{
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto   data = make_sparse_data(case_id, m, n, k, false);
        auto*  dA   = to_dev_tool(data.A);
        auto*  dB   = to_dev_tool(data.B);
        auto*  dC   = to_dev_tool(data.C);
        auto*  dI   = to_dev_tool(data.idxbuf);
        float* dD   = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, data.C.size() * sizeof(float)));
        if(m == 16)
            k_smf_16x16x32<E, Semantics::Float, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
        else
            k_smf_32x32x16<E, Semantics::Float, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, data.C.size());
        record_case(name,
                    case_id,
                    bits_of_vector(data.A),
                    bits_of_vector(data.B),
                    bits_of_vector(data.C),
                    bits_of_vector(data.idxbuf),
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dI);
        (void)hipFree(dD);
    }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8_16x16x64(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec      = fpsan::detail::v8_fragment<AE>;
    using BVec      = fpsan::detail::v16_fragment<BE>;
    constexpr int N = 16, K = 64, Cc = K / 2;
    const int     lane = threadIdx.x;
    const int     g    = lane / 16;
    const int     j    = lane % 16;
    AVec          an{};
    for(int b = 0; b < 8; ++b)
    {
        const int c = 16 * (b / 4) + 4 * g + (b % 4);
        an[b]       = AE(A[j * Cc + c]);
    }
    BVec bn{};
    for(int e = 0; e < 16; ++e)
    {
        const int kk = 32 * (e / 8) + 8 * g + (e % 8);
        bn[e]        = BE(B[kk * N + j]);
    }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * N + j];
    // CDNA3 fp8/bf8 SMFMAC builtins take A/B as packed i32 vectors. The local
    // fragment objects preserve byte order; bit-casting exposes the raw ABI.
    auto ai = __builtin_bit_cast(fpsan::v2i32_smf, an);
    auto bi = __builtin_bit_cast(fpsan::v4i32_smf, bn);
    auto d  = [&] {
        if constexpr(std::is_same_v<AE, fpsan::amd_fp8_e4m3>
                     && std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(ai, bi, cn, idx[lane], 0, 0);
        else if constexpr(std::is_same_v<AE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(ai, bi, cn, idx[lane], 0, 0);
        else if constexpr(std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(ai, bi, cn, idx[lane], 0, 0);
        else
            return __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(ai, bi, cn, idx[lane], 0, 0);
    }();
    for(int reg = 0; reg < 4; ++reg)
        D[(4 * g + reg) * N + j] = d[reg];
}

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8_32x32x32(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec      = fpsan::detail::v8_fragment<AE>;
    using BVec      = fpsan::detail::v16_fragment<BE>;
    constexpr int M = 32, N = 32, K = 32, Cc = K / 2;
    const int     lane = threadIdx.x;
    const int     g    = lane / 32;
    const int     j    = lane % 32;
    AVec          an{};
    for(int b = 0; b < 8; ++b)
    {
        const int c = 8 * (b / 4) + 4 * g + (b % 4);
        an[b]       = AE(A[j * Cc + c]);
    }
    const int jcol = (lane % 16) + 16 * ((lane / 16) % 2);
    const int kgrp = (lane / 16) / 2;
    BVec      bn{};
    for(int e = 0; e < 16; ++e)
    {
        const int kk = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
        bn[e]        = BE(B[kk * N + jcol]);
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < M; ++i)
        for(int jj = 0; jj < N; ++jj)
        {
            auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * N + jj];
        }
    // Same packed fp8/bf8 ABI as the 16x16x64 case, but with a larger C/D
    // accumulator fragment.
    auto ai = __builtin_bit_cast(fpsan::v2i32_smf, an);
    auto bi = __builtin_bit_cast(fpsan::v4i32_smf, bn);
    auto d  = [&] {
        if constexpr(std::is_same_v<AE, fpsan::amd_fp8_e4m3>
                     && std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(ai, bi, cn, idx[lane], 0, 0);
        else if constexpr(std::is_same_v<AE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(ai, bi, cn, idx[lane], 0, 0);
        else if constexpr(std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
            return __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(ai, bi, cn, idx[lane], 0, 0);
        else
            return __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(ai, bi, cn, idx[lane], 0, 0);
    }();
    for(int i = 0; i < M; ++i)
        for(int jj = 0; jj < N; ++jj)
        {
            auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
            if(loc.lane == lane)
                D[i * N + jj] = d[loc.reg];
        }
}

template <class AE, class BE>
void emit_smf_fp8(const char* name, int m, int k)
{
    const int n = m;
    for(int case_id = 0; case_id < kGoldenCaseCount; ++case_id)
    {
        auto   data = make_sparse_data(case_id, m, n, k, true);
        auto*  dA   = to_dev_tool(data.A);
        auto*  dB   = to_dev_tool(data.B);
        auto*  dC   = to_dev_tool(data.C);
        auto*  dI   = to_dev_tool(data.idxbuf);
        float* dD   = nullptr;
        HIP_CHECK_TOOL(hipMalloc(&dD, data.C.size() * sizeof(float)));
        if(m == 16)
            k_smf_fp8_16x16x64<AE, BE, Semantics::Float, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
        else
            k_smf_fp8_32x32x32<AE, BE, Semantics::Float, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
        HIP_CHECK_TOOL(hipDeviceSynchronize());
        auto got = from_dev_tool(dD, data.C.size());
        record_case(name,
                    case_id,
                    bits_of_vector(data.A),
                    bits_of_vector(data.B),
                    bits_of_vector(data.C),
                    bits_of_vector(data.idxbuf),
                    bits_of_vector(got));
        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dI);
        (void)hipFree(dD);
    }
}

void emit_header_begin(const char* arch_name)
{
    // The generated header is intentionally self-contained and has no dependency
    // on HIP or fpsan. Test files only need CaseView plus the raw bit arrays.
    std::cout << "// Copyright (c) 2026 Advanced Micro Devices, Inc.\n";
    std::cout << "// SPDX-License-Identifier: MIT\n";
    std::cout << "//\n";
    std::cout << "// Generated by fpsan_generate_goldens on " << arch_name
              << " using raw clang builtins.\n";
    std::cout << "// Do not edit fixture values by hand; regenerate on trusted silicon.\n";
    std::cout << "#ifndef FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_HPP\n";
    std::cout << "#define FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_HPP\n\n";
    std::cout << "#include <cstddef>\n#include <cstdint>\n\n";
    std::cout << "namespace fpsan_test\n{\nnamespace golden_cdna3\n{\n";
    std::cout << "    inline constexpr int kCaseCount = " << kGoldenCaseCount << ";\n\n";
    std::cout << "    struct CaseView\n    {\n";
    std::cout << "        const char*          name;\n";
    std::cout << "        int                  case_id;\n";
    std::cout << "        const std::uint64_t* a;\n";
    std::cout << "        std::size_t          a_count;\n";
    std::cout << "        const std::uint64_t* b;\n";
    std::cout << "        std::size_t          b_count;\n";
    std::cout << "        const std::uint64_t* c;\n";
    std::cout << "        std::size_t          c_count;\n";
    std::cout << "        const std::uint64_t* idx;\n";
    std::cout << "        std::size_t          idx_count;\n";
    std::cout << "        const std::uint64_t* d;\n";
    std::cout << "        std::size_t          d_count;\n";
    std::cout << "    };\n\n";
    std::cout << "    inline bool same_name(const char* lhs, const char* rhs)\n";
    std::cout << "    {\n";
    std::cout << "        while(*lhs && *rhs && *lhs == *rhs)\n";
    std::cout << "        {\n            ++lhs;\n            ++rhs;\n        }\n";
    std::cout << "        return *lhs == *rhs;\n";
    std::cout << "    }\n\n";
}

void emit_header_end()
{
    std::cout << "    inline constexpr CaseView kCases[] = {\n";
    for(const auto& c : g_cases)
    {
        std::cout << "        {\"" << c.name << "\", " << c.case_id << ", ";
        std::cout << c.a << ", " << c.a_count << ", ";
        std::cout << c.b << ", " << c.b_count << ", ";
        std::cout << c.c << ", " << c.c_count << ", ";
        if(c.idx.empty())
            std::cout << "nullptr, 0, ";
        else
            std::cout << c.idx << ", " << c.idx_count << ", ";
        std::cout << c.d << ", " << c.d_count << "},\n";
    }
    std::cout << "    };\n\n";
    std::cout << "    inline const CaseView* find_case(const char* name, int case_id)\n";
    std::cout << "    {\n";
    std::cout << "        for(const auto& c : kCases)\n";
    std::cout << "            if(c.case_id == case_id && same_name(c.name, name))\n";
    std::cout << "                return &c;\n";
    std::cout << "        return nullptr;\n";
    std::cout << "    }\n";
    std::cout << "} // namespace golden_cdna3\n";
    std::cout << "} // namespace fpsan_test\n\n";
    std::cout << "#endif // FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_HPP\n";
}

int main()
{
    int ndev = 0;
    HIP_CHECK_TOOL(hipGetDeviceCount(&ndev));
    if(ndev == 0)
    {
        std::cerr << "No HIP device available\n";
        return 1;
    }
    hipDeviceProp_t prop{};
    HIP_CHECK_TOOL(hipGetDeviceProperties(&prop, 0));
    std::string arch = prop.gcnArchName;
    if(arch.rfind("gfx94", 0) != 0)
    {
        std::cerr << "Refusing to generate CDNA3 goldens on non-gfx94x device: " << arch << "\n";
        return 1;
    }

    emit_header_begin(prop.gcnArchName);

    // Each registry macro expands the same declarative intrinsic list used by
    // the tests. Adding an intrinsic to the registry therefore creates both the
    // golden fixture and the consuming LayoutMatchesHardware test coverage.
#define EMIT_DENSE_VEC(Name, M_, N_, K_, Bk_, InBits_, AV_, BV_, CV_, WRAP_) \
    emit_dense_vec<Name>(#Name);
    FPSAN_CDNA3_DENSE_VEC_MATRIX_INTRINSICS(EMIT_DENSE_VEC)
#undef EMIT_DENSE_VEC

#define EMIT_DENSE_F32(Name, M_, N_, K_, Bk_, CVec_, WRAP_) emit_dense_f32<Name>(#Name);
    FPSAN_CDNA3_DENSE_F32_MATRIX_INTRINSICS(EMIT_DENSE_F32)
#undef EMIT_DENSE_F32

    emit_f64_16();
    emit_f64_4();

#define EMIT_SMF_H(Name, Elem_, M_, N_, K_) emit_smf_h<Elem_>(#Name, M_, N_, K_);
    FPSAN_CDNA3_SMF_H_MATRIX_INTRINSICS(EMIT_SMF_H)
#undef EMIT_SMF_H

#define EMIT_SMF_FP8(Name, AElem_, BElem_, M_, K_) emit_smf_fp8<AElem_, BElem_>(#Name, M_, K_);
    FPSAN_CDNA3_SMF_FP8_MATRIX_INTRINSICS(EMIT_SMF_FP8)
#undef EMIT_SMF_FP8

    emit_header_end();
    return 0;
}
