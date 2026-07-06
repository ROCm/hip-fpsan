// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_wave.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU wave-cooperative intrinsics (wave reductions,
// cross-lane data movers, ...). Like amdgcn_matrix.hpp, this is GPU-only and
// opt-in (not pulled by <fpsan/fpsan.hpp>).
//
// The wave-reduce family relies on the selected Value<> operation being
// order-independent for the certified reductions:
//   - FPSan addition uses the mode's payload addition (Triton's Z/2^w ring or
//     the algebraic Z/nZ variants), so a butterfly reduce produces the same
//     payload as a sequential reduce once the lane exchange covers the full
//     wave. Strategy and tree shape are irrelevant in FPSan mode.
//   - FPSan fmin/fmax use fpsan::min/max. Triton keeps signed-payload order;
//     algebraic semantics use qr-positive order with deterministic same-class
//     tie-breaking. The same butterfly works because the selected min/max
//     operation is associative and commutative.
//
// The wrappers are emitted by the FPSAN_DEFINE_WAVE_REDUCE macro; new (op,
// type) pairs are one-liners.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_WAVE_HPP
#define FPSAN_AMDGCN_WAVE_HPP

#include "fpsan/amdgcn_matrix.hpp" // for detail::wave_lane + detail::wave_shfl
#include "fpsan/math.hpp"          // for fpsan::min / fpsan::max
#include "fpsan/value.hpp"

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_wave.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan {

// Define one fpsan::<name> wave-reduce wrapper. Native mode forwards to BUILTIN
// (passes Strategy as the _Constant int32_t arg). FPSan mode runs a wave-size
// correct XOR butterfly with COMBINE_EXPR, where the names `r` and `other` are
// in scope and resolve to per-stage Values. Strategy is ignored in FPSan mode
// (for fadd/fmin/fmax: the selected Value<> combine op is associative and
// commutative, so the tree shape doesn't matter; for fsub: we pick the same
// butterfly shape as the rest of the family and document that the Float and
// FPSan paths may differ in last-stage rounding -- the FPSan answer is the
// one that matches an independent host scalar butterfly reference, which is
// the only thing FPSan is asked to certify).
#define FPSAN_DEFINE_WAVE_REDUCE(name, type, COMBINE_EXPR, BUILTIN)                                \
  template <int Strategy = 0, Semantics S = Semantics::Native,                                     \
            Conversions C = Conversions::Explicit>                                                 \
  FPSAN_DEVICE Value<type, S, C> name(Value<type, S, C> v) {                                       \
    if constexpr (S == Semantics::Native) {                                                        \
      return Value<type, S, C>(BUILTIN(v.to_float(), Strategy));                                   \
    } else {                                                                                       \
      /* Wave-size-correct XOR butterfly. detail::wave_shfl handles gfx11                          \
       * wave64 cross-half moves with V_PERMLANE64_B32 before same-half DS                         \
       * bpermute. The gfx11 wave64 Float builtin's default LLVM lowering is                       \
       * tracked separately in the coverage docs because it does not match                         \
       * LLVM's documented full-wave reduction contract on tested silicon. */                      \
      const int ws = __builtin_amdgcn_wavefrontsize();                                             \
      const int lane = detail::wave_lane_full();                                                   \
      Value<type, S, C> r = v;                                                                     \
      for (int off = 1; off < ws; off <<= 1) {                                                     \
        auto other = detail::wave_shfl(r, lane ^ off);                                             \
        r = (COMBINE_EXPR);                                                                        \
      }                                                                                            \
      return r;                                                                                    \
    }                                                                                              \
  }

// ---- f32 ---- (gfx10+: __builtin_amdgcn_wave_reduce_f* on RDNA)
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wave_reduce_fadd_f32)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fadd_f32, float, r + other,
                         __builtin_amdgcn_wave_reduce_fadd_f32)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fsub_f32, float, r - other,
                         __builtin_amdgcn_wave_reduce_fsub_f32)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fmin_f32, float, fpsan::min(r, other),
                         __builtin_amdgcn_wave_reduce_fmin_f32)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fmax_f32, float, fpsan::max(r, other),
                         __builtin_amdgcn_wave_reduce_fmax_f32)
#endif

// ---- f64 ----
// Visible on several targets, but direct probes only lowered cleanly for gfx11
// and gfx12 in the current audited set. Keep these wrappers off gfx94x/gfx950 so
// those targets do not accidentally instantiate a Float path that their backends
// still reject.
#if !defined(__HIP_DEVICE_COMPILE__) || ((defined(__GFX11__) || defined(__GFX12__)) &&             \
                                         __has_builtin(__builtin_amdgcn_wave_reduce_fadd_f64))
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fadd_f64, double, r + other,
                         __builtin_amdgcn_wave_reduce_fadd_f64)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fsub_f64, double, r - other,
                         __builtin_amdgcn_wave_reduce_fsub_f64)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fmin_f64, double, fpsan::min(r, other),
                         __builtin_amdgcn_wave_reduce_fmin_f64)
FPSAN_DEFINE_WAVE_REDUCE(amdgcn_wave_reduce_fmax_f64, double, fpsan::max(r, other),
                         __builtin_amdgcn_wave_reduce_fmax_f64)
#endif
#undef FPSAN_DEFINE_WAVE_REDUCE

// =============================================================================
// Cross-lane data movers (readlane / readfirstlane / writelane / ds_bpermute /
// ds_permute / ds_swizzle / mov_dpp / update_dpp / permlane*).
//
// At the builtin level these all operate on i32 (or i64 for f64 storage); for
// FPSan they ARE pure bit movement -- the lane's payload (FPSan mode) or float
// bits (Native mode) is moved verbatim across lanes. Native-mode and FPSan-mode
// share the same implementation (bit-cast storage -> apply builtin -> bit-cast
// back). That's the whole identity the wrappers exploit: cross-lane moves
// don't observe values, just bits.
// =============================================================================

namespace detail {

// Apply a 32-bit cross-lane operation to the storage bits of v and rebuild the
// Value. For 64-bit scalars (Value<double>), run the op on the lo/hi 32-bit
// halves and reassemble. `Op` is callable as op(std::uint32_t) ->
// std::uint32_t.
template <class FT, Semantics S, Conversions C, class Op>
FPSAN_DEVICE Value<FT, S, C> bit_move(Value<FT, S, C> v, Op op) {
  using B = typename Value<FT, S, C>::bits_type;
  static_assert(!Value<FT, S, C>::is_vector, "bit_move requires a scalar Value");
  const auto bits = v.to_storage_bits();
  if constexpr (sizeof(B) <= 4) {
    const std::uint32_t w = static_cast<std::uint32_t>(bits);
    const std::uint32_t got = op(w);
    return Value<FT, S, C>::from_storage_bits(static_cast<B>(got));
  } else {
    static_assert(sizeof(B) == 8, "bit_move supports up to 8-byte scalars (f64)");
    const std::uint64_t b64 = static_cast<std::uint64_t>(bits);
    const std::uint32_t lo = static_cast<std::uint32_t>(b64);
    const std::uint32_t hi = static_cast<std::uint32_t>(b64 >> 32);
    const std::uint32_t glo = op(lo);
    const std::uint32_t ghi = op(hi);
    const std::uint64_t g64 =
        (static_cast<std::uint64_t>(ghi) << 32) | static_cast<std::uint64_t>(glo);
    return Value<FT, S, C>::from_storage_bits(static_cast<B>(g64));
  }
}

} // namespace detail

// ---- readlane / readfirstlane (broadcast from a chosen lane) ----------------
// The `lane` argument must be wave-uniform; behavior is undefined otherwise --
// see the LLVM intrinsic doc. Our wrapper passes it through unchanged.
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_readlane(Value<FT, S, C> v, int lane) {
  return detail::bit_move(v, [lane](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(w), lane));
  });
}

template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_readfirstlane(Value<FT, S, C> v) {
  return detail::bit_move(v, [](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(w)));
  });
}

// Note: __builtin_amdgcn_writelane is NOT exposed by Clang as a builtin (it is
// accessible via HIP runtime helpers and the LLVM IR intrinsic, but there is
// no Clang builtin for it). Customers needing writelane on FPSan Values can
// bit-cast via to_storage_bits()/from_storage_bits() and call the HIP
// runtime's __ockl_writelane_*; we omit a fpsan:: wrapper rather than pull in
// the HIP runtime.

// ---- ds_bpermute / ds_permute (indexed cross-lane gather/scatter) -----------
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_ds_bpermute(int addr, Value<FT, S, C> v) {
  return detail::bit_move(v, [addr](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(__builtin_amdgcn_ds_bpermute(addr, static_cast<int>(w)));
  });
}

template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_ds_permute(int addr, Value<FT, S, C> v) {
  return detail::bit_move(v, [addr](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(__builtin_amdgcn_ds_permute(addr, static_cast<int>(w)));
  });
}

// ---- ds_swizzle (fixed permutation; pattern is a compile-time constant) -----
template <int Pattern, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_ds_swizzle(Value<FT, S, C> v) {
  return detail::bit_move(v, [](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(w), Pattern));
  });
}

// ---- mov_dpp / update_dpp (data parallel primitives) ------------------------
// Ctrl/RowMask/BankMask/BoundCtrl must be compile-time constants. update_dpp
// blends the result with `old` according to row_mask/bank_mask/bound_ctrl.
template <int Ctrl, int RowMask, int BankMask, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_mov_dpp(Value<FT, S, C> v) {
  return detail::bit_move(v, [](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_mov_dpp(static_cast<int>(w), Ctrl, RowMask, BankMask, BoundCtrl));
  });
}

template <int Ctrl, int RowMask, int BankMask, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_update_dpp(Value<FT, S, C> old, Value<FT, S, C> v) {
  using B = typename Value<FT, S, C>::bits_type;
  static_assert(!Value<FT, S, C>::is_vector, "amdgcn_update_dpp requires a scalar Value");
  const auto vbits = v.to_storage_bits();
  const auto obits = old.to_storage_bits();
  if constexpr (sizeof(B) <= 4) {
    const std::uint32_t w = static_cast<std::uint32_t>(vbits);
    const std::uint32_t o = static_cast<std::uint32_t>(obits);
    const std::uint32_t got = static_cast<std::uint32_t>(__builtin_amdgcn_update_dpp(
        static_cast<int>(o), static_cast<int>(w), Ctrl, RowMask, BankMask, BoundCtrl));
    return Value<FT, S, C>::from_storage_bits(static_cast<B>(got));
  } else {
    static_assert(sizeof(B) == 8);
    const std::uint64_t vb = static_cast<std::uint64_t>(vbits);
    const std::uint64_t ob = static_cast<std::uint64_t>(obits);
    const std::uint32_t vlo = static_cast<std::uint32_t>(vb);
    const std::uint32_t vhi = static_cast<std::uint32_t>(vb >> 32);
    const std::uint32_t olo = static_cast<std::uint32_t>(ob);
    const std::uint32_t ohi = static_cast<std::uint32_t>(ob >> 32);
    const std::uint32_t glo = static_cast<std::uint32_t>(__builtin_amdgcn_update_dpp(
        static_cast<int>(olo), static_cast<int>(vlo), Ctrl, RowMask, BankMask, BoundCtrl));
    const std::uint32_t ghi = static_cast<std::uint32_t>(__builtin_amdgcn_update_dpp(
        static_cast<int>(ohi), static_cast<int>(vhi), Ctrl, RowMask, BankMask, BoundCtrl));
    const std::uint64_t g64 =
        (static_cast<std::uint64_t>(ghi) << 32) | static_cast<std::uint64_t>(glo);
    return Value<FT, S, C>::from_storage_bits(static_cast<B>(g64));
  }
}

// ---- permlane16 / permlanex16 / permlane64 ----------------------------------
// permlane16(old, src, sel0, sel1, fi, bc): permute within rows of 16. sel0
// and sel1 are 32-bit lane-selector vectors (one nibble per lane); fi (fetch
// invalid), bc (bound control) are bools. gfx10+ (not on CDNA gfx9 family).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane16)
template <bool FetchInvalid, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane16(Value<FT, S, C> old, Value<FT, S, C> src, int sel0,
                                               int sel1) {
  return detail::bit_move(src, [&](std::uint32_t w) -> std::uint32_t {
    const std::uint32_t o = static_cast<std::uint32_t>(old.to_storage_bits());
    return static_cast<std::uint32_t>(__builtin_amdgcn_permlane16(
        static_cast<int>(o), static_cast<int>(w), static_cast<unsigned>(sel0),
        static_cast<unsigned>(sel1), FetchInvalid, BoundCtrl));
  });
}

template <bool FetchInvalid, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlanex16(Value<FT, S, C> old, Value<FT, S, C> src, int sel0,
                                                int sel1) {
  return detail::bit_move(src, [&](std::uint32_t w) -> std::uint32_t {
    const std::uint32_t o = static_cast<std::uint32_t>(old.to_storage_bits());
    return static_cast<std::uint32_t>(__builtin_amdgcn_permlanex16(
        static_cast<int>(o), static_cast<int>(w), static_cast<unsigned>(sel0),
        static_cast<unsigned>(sel1), FetchInvalid, BoundCtrl));
  });
}
#endif // __has_builtin(__builtin_amdgcn_permlane16)

// ---- permlane64: swap lane i with lane i+32 within a wave64. On wave32 (the
// default RDNA4 mode) lanes 32..63 don't exist, so the swap has no observable
// effect at the wave32 level; the wrapper is provided for completeness when
// customers run wave64. gfx11+ only.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane64)
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane64(Value<FT, S, C> v) {
  // LLVM documents llvm.amdgcn.permlane64 as a wave64 half-swap that is a
  // no-op in wave32 mode. Some gfx12 hardware/compiler combinations still
  // produce surprising results for small integer payloads, so enforce the
  // documented wave32 behavior before exposing it through FPSan storage bits.
  if (__builtin_amdgcn_wavefrontsize() == 32)
    return v;
  return detail::bit_move(
      v, [](std::uint32_t w) -> std::uint32_t { return __builtin_amdgcn_permlane64(w); });
}
#endif

// ---- gfx1250 permlane data movers: bcast / down / up / xor (b32) ------------
// v_permlane_{bcast,down,up,xor}_b32 vdst, vsrc, ssrc0, ssrc1 -- cross-lane move
// of the data payload, governed by two wave-uniform scalar selectors. These are
// pure bit movement (they relocate a lane's storage, never observing the value),
// so Float and FPSan modes share one implementation. `sel0`/`sel1` MUST be
// wave-uniform (they lower to SGPR operands); behavior is undefined otherwise.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane_bcast)
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane_bcast(Value<FT, S, C> v, int sel0, int sel1) {
  return detail::bit_move(v, [&](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_permlane_bcast(static_cast<int>(w), sel0, sel1));
  });
}
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane_down(Value<FT, S, C> v, int sel0, int sel1) {
  return detail::bit_move(v, [&](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_permlane_down(static_cast<int>(w), sel0, sel1));
  });
}
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane_up(Value<FT, S, C> v, int sel0, int sel1) {
  return detail::bit_move(v, [&](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_permlane_up(static_cast<int>(w), sel0, sel1));
  });
}
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_permlane_xor(Value<FT, S, C> v, int sel0, int sel1) {
  return detail::bit_move(v, [&](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_permlane_xor(static_cast<int>(w), sel0, sel1));
  });
}
#endif // __has_builtin(__builtin_amdgcn_permlane_bcast)

// ---- permlane_idx_gen: v_permlane_idx_gen_b32 vdst, vsrc, ssrc0 -------------
// Generates a per-lane permute INDEX (an integer to feed a subsequent permute),
// not a moved float payload, so it is exposed as a plain integer helper. Being
// pure integer index arithmetic, it is mode-independent (no Float/FPSan split).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane_idx_gen)
FPSAN_DEVICE inline std::uint32_t amdgcn_permlane_idx_gen(int src, int sel) {
  return static_cast<std::uint32_t>(__builtin_amdgcn_permlane_idx_gen(src, sel));
}
#endif

// ---- ds_bpermute_fi_b32: ds_bpermute with FI (fetch-invalid) support --------
// Same indexed cross-lane gather as ds_bpermute (addr selects the source lane),
// with the gfx1250 fetch-invalid behavior; pure bit movement.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_bpermute_fi_b32)
template <class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_ds_bpermute_fi(int addr, Value<FT, S, C> v) {
  return detail::bit_move(v, [addr](std::uint32_t w) -> std::uint32_t {
    return static_cast<std::uint32_t>(
        __builtin_amdgcn_ds_bpermute_fi_b32(addr, static_cast<int>(w)));
  });
}
#endif

// ---- permlane16_swap: swap two registers' payloads across lane pairs --------
// v_permlane16_swap_b32 returns {x', y'} (a uint2); it swaps the two operands'
// lanes within rows of 16. FetchInvalid / BoundCtrl are compile-time immediates.
// Pure bit movement on both operands; updates x and y in place. Scalar 32-bit
// storage only.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane16_swap)
template <bool FetchInvalid, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE void amdgcn_permlane16_swap(Value<FT, S, C> &x, Value<FT, S, C> &y) {
  using B = typename Value<FT, S, C>::bits_type;
  static_assert(!Value<FT, S, C>::is_vector && sizeof(B) <= 4,
                "amdgcn_permlane16_swap requires a scalar 32-bit Value");
  const std::uint32_t xw = static_cast<std::uint32_t>(x.to_storage_bits());
  const std::uint32_t yw = static_cast<std::uint32_t>(y.to_storage_bits());
  auto r = __builtin_amdgcn_permlane16_swap(static_cast<int>(xw), static_cast<int>(yw),
                                            FetchInvalid, BoundCtrl);
  x = Value<FT, S, C>::from_storage_bits(static_cast<B>(static_cast<std::uint32_t>(r[0])));
  y = Value<FT, S, C>::from_storage_bits(static_cast<B>(static_cast<std::uint32_t>(r[1])));
}
#endif

// ---- permlane32_swap: swap two registers' payloads across 32-lane halves ----
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane32_swap)
template <bool FetchInvalid, bool BoundCtrl, class FT, Semantics S, Conversions C>
FPSAN_DEVICE void amdgcn_permlane32_swap(Value<FT, S, C> &x, Value<FT, S, C> &y) {
  using B = typename Value<FT, S, C>::bits_type;
  static_assert(!Value<FT, S, C>::is_vector && sizeof(B) <= 4,
                "amdgcn_permlane32_swap requires a scalar 32-bit Value");
  const std::uint32_t xw = static_cast<std::uint32_t>(x.to_storage_bits());
  const std::uint32_t yw = static_cast<std::uint32_t>(y.to_storage_bits());
  auto r = __builtin_amdgcn_permlane32_swap(static_cast<int>(xw), static_cast<int>(yw),
                                            FetchInvalid, BoundCtrl);
  x = Value<FT, S, C>::from_storage_bits(static_cast<B>(static_cast<std::uint32_t>(r[0])));
  y = Value<FT, S, C>::from_storage_bits(static_cast<B>(static_cast<std::uint32_t>(r[1])));
}
#endif

// ---- mov_dpp8: DPP variant with a per-lane selector encoded as a 32-bit
// immediate (8 nibbles, one per lane within an 8-lane row). gfx10+ only.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mov_dpp8)
template <unsigned Sel, class FT, Semantics S, Conversions C>
FPSAN_DEVICE Value<FT, S, C> amdgcn_mov_dpp8(Value<FT, S, C> v) {
  return detail::bit_move(
      v, [](std::uint32_t w) -> std::uint32_t { return __builtin_amdgcn_mov_dpp8(w, Sel); });
}
#endif

// ---- ballot: wave-wide bool ballot. The bool itself may come from any
// source (an FPSan compare on payloads, a user predicate, ...) -- ballot
// itself just collects one bit per active lane, so it has identical behavior
// in Float and FPSan modes. w32 only on wave32 archs; w64 on every arch with
// a 64-lane wave.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ballot_w32)
FPSAN_DEVICE inline std::uint32_t amdgcn_ballot_w32(bool b) {
  return static_cast<std::uint32_t>(__builtin_amdgcn_ballot_w32(b));
}
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ballot_w64)
FPSAN_DEVICE inline std::uint64_t amdgcn_ballot_w64(bool b) {
  return static_cast<std::uint64_t>(__builtin_amdgcn_ballot_w64(b));
}
#endif

} // namespace fpsan

#endif // FPSAN_AMDGCN_WAVE_HPP
