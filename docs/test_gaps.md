# Test Gaps

Last scanned: 2026-06-10.

This document tracks test limitations that matter for ISA conformance and
emulator/silicon validation. The main distinction is:

- **Behavioral oracle**: a test can catch a buggy hardware or emulator
  implementation because it compares the instruction result with an independent
  scalar, host, spec-derived, or separately implemented reference.
- **Mapping/parity test**: a test proves the wrapper calls the intended builtin
  or has the same observable result as a direct builtin call. If the underlying
  instruction is implemented incorrectly in both paths, the test can still pass.
- **FPSan-model test**: a test validates the FPSan payload-domain replacement
  for an intrinsic. These are useful for wrapper correctness, but usually do
  not execute the hardware instruction in FPSan mode.

The scan below covers the AMDGPU intrinsic tests across the test tree, not just
the CDNA3 additions.

Architecture scope labels used below:

- **Host/generic**: CPU or generic FPSan tests; not an AMDGPU ISA test.
- **HIP portable**: registered for every HIP-enabled build in this project.
- **Shared gfx12/gfx94x**: registered when either a gfx12 target or a gfx94x
  CDNA3 target is present.
- **Shared gfx12/gfx94x/gfx950**: registered for gfx12, gfx94x CDNA3, or gfx950
  CDNA4 builds.
- **gfx12**: RDNA4/gfx12-specific test registration.
- **gfx94x/CDNA3**: gfx940, gfx941, or gfx942-specific test registration.
- **gfx950/CDNA4**: gfx950-specific test registration.

## Suite classification

| Test file | Scope | Area | Oracle quality |
| --- | --- | --- | --- |
| `amdgcn_math_test.cpp` | Shared gfx12/gfx94x/gfx950, with gated gfx12/CDNA3 dot cases | f32/f64/f16 scalar math, `fdot2`, gfx12 dot variants | Float mode is direct-builtin parity. FPSan mode checks wrapper algebra. |
| `amdgcn_ldexp_test.cpp` | Shared gfx12/gfx94x/gfx950 | `ldexpf`, `ldexp`, `ldexph` | Behavioral oracle against host `std::ldexp` on exact inputs. |
| `cvt_test.cpp` | Shared gfx12/gfx94x | baseline `pkrtz`, fp8/bf8 pack/unpack | `pkrtz` has a host oracle on exact f16 inputs. Other Float fp8/bf8 paths are mostly direct-builtin parity. |
| `cvt_gfx950_test.cpp` | gfx950/CDNA4 | non-scaled fp8/bf8 conversions | Mostly behavioral: all-byte unpack checks and round-trip/host references. |
| `cvt_scalef32_*_gfx950_test.cpp` | gfx950/CDNA4 | scaled fp8/fp6/bf6/fp4 conversions | Strong behavioral coverage against host OCP narrow/widen and independent payload references. |
| `global_load_tr_test.cpp` | gfx12 | global transposed loads | Direct-builtin parity only. Good wrapper check, not an independent transpose oracle. |
| `ds_read_tr_test.cpp` | gfx950/CDNA4 | LDS transposed reads | Direct-builtin parity only. Good wrapper check, not an independent transpose oracle. |
| `xlane_test.cpp` | gfx12 | cross-lane data movers | Mixed: readlane, readfirstlane, `ds_{b,}permute`, identity DPP, and ballot w32 have host mapping oracles; `ds_swizzle`, `permlane64`, and untested movers remain weak. |
| `wave_test.cpp` | Shared gfx12/gfx94x/gfx950 | f32 wave reductions | Behavioral for fadd/fmin/fmax over runtime wave size. fsub has only FPSan butterfly coverage; Float oracle is skipped. |
| `classify_test.cpp` | HIP portable | `classf`, `fcmpf` | Weak: `classf` uses all-categories mask on finite inputs; `fcmpf` only checks Float/FPSan agreement. |
| `atomic_test.cpp` | HIP portable | f32/f64 and packed f16/bf16 atomics | Behavioral final-value and return-value references for exact/order-independent inputs. |
| `wmma_test.cpp` | gfx12 | WMMA | Strong layout/dataflow oracles plus FPSan scalar references for tested exact-input domains. |
| `swmmac_gfx12_test.cpp` | gfx12 | sparse WMMA | Direct-builtin smoke plus stronger layout/dataflow and FPSan scalar-reference tests. |
| `mfma_cdna3_test.cpp`, `smfmac_cdna3_test.cpp` | gfx94x/CDNA3 | MFMA/SMFMAC | Strong layout/dataflow and FPSan scalar-reference tests for tested exact-input domains. |
| `mfma_test.cpp` | gfx950/CDNA4 | MFMA/SMFMAC/scaled matrix | Mostly strong; a few mixed sub-byte scaled combinations are one-sided. |
| `math_test.cpp` | Host/generic | generic FPSan math | Not a GPU intrinsic test. Do not count it as `__builtin_amdgcn_*` coverage. |

## High-priority gaps

### AMDGCN scalar math tests are mostly mapping tests

The scalar math tests compare Float-mode wrapper output with the direct
`__builtin_amdgcn_*` result in the same kernel. This validates wrapper plumbing,
but a buggy emulator implementation of the underlying instruction can still
pass.

Mapping-only Float coverage exists for these architecture scopes:

- **Shared gfx12/gfx94x/gfx950**: f32 `amdgcn_rcpf`, `amdgcn_sqrtf`,
  `amdgcn_rsqf`, `amdgcn_rsq_clampf`, `amdgcn_sinf`, `amdgcn_cosf`,
  `amdgcn_logf`, `amdgcn_exp2f`, `amdgcn_fractf`, `amdgcn_fmed3f`; f64
  `amdgcn_rcp`, `amdgcn_sqrt`, `amdgcn_rsq`, `amdgcn_rsq_clamp`,
  `amdgcn_fract`; and f16 `amdgcn_rcph`, `amdgcn_sqrth`, `amdgcn_rsqh`,
  `amdgcn_sinh`, `amdgcn_cosh`, `amdgcn_fracth`, `amdgcn_fmed3h`.

Wrappers that currently lack direct meaningful tests in any scanned suite:

- `amdgcn_log_clampf`
- `amdgcn_tanhf`
- `amdgcn_tanhh`

For the untested list, target availability is wrapper/builtin-specific.
`log_clampf` is visible to `__has_builtin` on gfx942 but rejects in direct
lowering probes; `tanhf`/`tanhh` need the `tanh-insts` feature in the audited
toolchain/targets. Keep them out of runtime tests until direct compile probes
lower cleanly on the intended targets.

Closure:

- Add behavioral tests for exact or well-specified subsets: `sqrt*`, `fract*`,
  and `fmed3*` can use straightforward host references on finite exact inputs.
- For approximate/transcendental instructions (`rcp*`, `rsq*`, `sin*`, `cos*`,
  `log*`, `exp2*`, `tanh*`), define the oracle policy first: ISA-specified
  tolerance, known-good silicon vectors, or documented ULP/relative-error
  limits.
- Keep direct-builtin parity tests too; they still prove the wrapper lowers to
  the intended builtin.

### Baseline fp8/bf8 conversions are mixed and architecture-specific

`cvt_test.cpp` is **shared gfx12/gfx94x** coverage and covers:

- `amdgcn_cvt_pkrtz`: behavioral on exact f16-representable inputs.
- `amdgcn_cvt_f32_fp8` and `amdgcn_cvt_f32_bf8`: Float mode compares wrapper
  to direct builtin, so a buggy unpack instruction can pass.
- `amdgcn_cvt_pk_fp8_f32` and `amdgcn_cvt_pk_bf8_f32`: Float mode compares
  wrapper to direct builtin, so a buggy pack instruction can pass.
- `amdgcn_cvt_pk_f32_fp8` and `amdgcn_cvt_pk_f32_bf8`: shared mapping/FPSan
  plumbing tests cover word selection and lane payload placement, but Float
  mode is still direct-builtin parity.
- `amdgcn_cvt_sr_fp8_f32` and `amdgcn_cvt_sr_bf8_f32`: shared mapping/FPSan
  plumbing tests cover byte selection, old-word splicing, and seed opacity on
  exact inputs, but they are not numeric stochastic-rounding oracles.

`cvt_gfx950_test.cpp` adds stronger behavioral **gfx950/CDNA4-only** coverage
for the latter four wrappers:

- `amdgcn_cvt_pk_f32_fp8`
- `amdgcn_cvt_pk_f32_bf8`
- `amdgcn_cvt_sr_fp8_f32`
- `amdgcn_cvt_sr_bf8_f32`

Those gfx950 tests use all-byte host OCP decode checks, exact-input round trips,
and byte/word placement invariants. That coverage does not automatically cover
gfx12 or gfx94x behavior, especially where the FP8/BF8 encodings differ.

Semantic-equivalence note from the RDNA4/CDNA4 doc audit: the baseline
conversion mnemonic names are shared, but the numeric formats are not uniformly
architecture-independent. CDNA3 documents AMD FNUZ FP8/BF8 encodings, CDNA4
documents OCP FP8/BF8 encodings, and RDNA4 documents both bias modes. Therefore
`cvt_test.cpp` is valid shared wrapper/plumbing coverage for builtin forwarding,
byte/word selection, and FPSan payload splicing, but it must not be counted as a
shared numeric FP8/BF8 oracle unless the expected encoding is selected per
target.

Additional **gfx950/CDNA4** conversion wrappers with no direct test found in the
scanned suite:

- `amdgcn_cvt_sr_f16_f32`
- `amdgcn_cvt_sr_bf16_f32`

Closure:

- Add gfx12/gfx94x behavioral tests for `amdgcn_cvt_pk_f32_{fp8,bf8}` and
  `amdgcn_cvt_sr_{fp8,bf8}_f32`, using the target's expected FP8/BF8 encoding.
- Replace or supplement baseline `cvt_f32_{fp8,bf8}` and
  `cvt_pk_{fp8,bf8}_f32` Float parity checks with host byte-pattern and exact
  pack references.
- Add `cvt_sr_{f16,bf16}_f32` tests that verify lane selection, old-lane
  preservation, deterministic exact inputs, and seed opacity where exact inputs
  make SR irrelevant.

### Dot-product tests are mapping tests for Float mode

The dot-product wrappers currently compare Float-mode output with direct
builtins. That validates lowering but not instruction behavior.

Affected wrappers by current test scope:

- **gfx12 and gfx94x/CDNA3**: `amdgcn_fdot2`.
- **gfx12 only**: `amdgcn_fdot2_f16_f16`,
  `amdgcn_fdot2_f32_bf16`, `amdgcn_dot4_f32_fp8_fp8`,
  `amdgcn_dot4_f32_fp8_bf8`, `amdgcn_dot4_f32_bf8_fp8`, and
  `amdgcn_dot4_f32_bf8_bf8`.

There is no current gfx94x/CDNA3 dot4 behavioral or mapping test in the scanned
suite.

Closure:

- Add Float-mode scalar references for exact-valued input sets.
- Include accumulator edge cases and mixed-sign inputs.
- For FP8/BF8 dot4, cover known byte patterns across normal, subnormal, zero,
  NaN, and architecture-specific FP8/BF8 encodings.

### Transposed load/read tests are direct-builtin parity only

The transposed data-movement tests are useful wrapper checks, but they are not
independent oracles for the transpose instructions themselves:

- **gfx12**: `amdgcn_global_load_tr_b128_f16` and
  `amdgcn_global_load_tr_b128_bf16`.
- **gfx950/CDNA4**: `amdgcn_ds_read_tr16_b64_f16`,
  `amdgcn_ds_read_tr16_b64_bf16`, `amdgcn_ds_read_tr8_b64_fp8`,
  `amdgcn_ds_read_tr8_b64_bf8`, `amdgcn_ds_read_tr4_b64`, and
  `amdgcn_ds_read_tr6_b96`.

If an emulator implements the transpose lane mapping incorrectly, the raw
builtin path and wrapper path can agree and still be wrong.

Closure:

- Encode the expected lane/slot transpose mapping as a host reference, either
  from ISA documentation or from separately documented silicon reverse
  engineering.
- Keep the current parity tests as wrapper-lowering checks.
- Add nontrivial per-lane/per-slot patterns that can detect row/column swaps,
  off-by-one lane groups, and packed sub-byte bitfield mistakes.

## Medium-priority gaps

### Classification and compare coverage is weak

Current classify/compare coverage is **HIP portable** and checks useful
invariants, but not full instruction behavior:

- `amdgcn_classf` uses mask `0x3FF` on finite inputs, so the expected answer is
  always true.
- `amdgcn_fcmpf` checks only Float/FPSan agreement for one predicate.
- `amdgcn_class`, `amdgcn_classh`, and `amdgcn_fcmp` have no direct tests in
  any scanned suite.

Closure:

- Add class tests for every category mask using explicit bit patterns:
  positive/negative zero, subnormal, normal, infinity, quiet NaN, and signaling
  NaN where representable.
- Add compare tests for multiple predicates with an independent host mask
  reference, including unordered NaN cases.
- Extend the same coverage to f64 and f16 wrappers where the builtins lower on
  the target.

### Wave and cross-lane coverage is incomplete

Strong behavioral coverage exists for these scopes:

- **Shared gfx12/gfx94x/gfx950**: `amdgcn_wave_reduce_fadd_f32`,
  `amdgcn_wave_reduce_fmin_f32`, and `amdgcn_wave_reduce_fmax_f32`.
- **gfx12 only**: `amdgcn_readlane`, `amdgcn_readfirstlane`,
  `amdgcn_ds_bpermute`, `amdgcn_ds_permute`, identity `amdgcn_mov_dpp`,
  identity `amdgcn_mov_dpp8`, and `amdgcn_ballot_w32`.

Remaining gaps:

- **Shared gfx12/gfx94x/gfx950**: `amdgcn_wave_reduce_fsub_f32` skips the
  Float host-reference test because subtraction is order-sensitive and the
  hardware reduction order is not encoded as a stable host oracle.
- **gfx12 only**: `amdgcn_ds_swizzle` checks only Float/FPSan lane-mapping
  agreement for two patterns, not the ISA mapping independently.
- **gfx12 only**: `amdgcn_mov_dpp` and `amdgcn_mov_dpp8` use identity patterns
  only.
- **gfx12 only in the current suite**: `amdgcn_permlane64` is currently a
  wave32 smoke/identity test; it does not validate wave64 half-swap behavior.
- `amdgcn_update_dpp`, `amdgcn_permlane16`, `amdgcn_permlanex16`, and
  `amdgcn_ballot_w64` have no direct tests in any scanned suite.

Closure:

- Add host mapping references for representative non-identity DPP and swizzle
  patterns.
- Add wave64-specific tests for `permlane64` and `ballot_w64` on targets that
  support those modes.
- Add update/permlane16/permlanex16 tests that cover old-value blending,
  selector interpretation, row/bank masks, and bound-control behavior.

### Matrix coverage is strong but not exhaustive

WMMA, MFMA, SMFMAC, and SWMMAC have strong behavioral coverage in the main
layout/dataflow tests. The architecture split is:

- **gfx12**: `wmma_test.cpp` and `swmmac_gfx12_test.cpp`.
- **gfx94x/CDNA3**: `mfma_cdna3_test.cpp` and `smfmac_cdna3_test.cpp`.
- **gfx950/CDNA4**: `mfma_test.cpp`.

These tests compare hardware instructions against independent software or host
scalar references on exact small inputs, then separately check FPSan payload
algebra.

Remaining limitations:

- Inputs intentionally use small exact values; the tests do not broadly exercise
  rounding, overflow, denormal/flush, signed-zero, infinity, or NaN behavior.
- xF32 coverage is useful for layout and gross arithmetic behavior, but it is
  not broad mantissa-rounding validation.
- FP8/BF8 matrix tests exercise important formats and layouts, but not
  exhaustive byte-pattern behavior inside every matrix instruction.
- In **gfx950/CDNA4** `mfma_test.cpp`, a few mixed sub-byte scaled combinations
  are one-sided: `ScaledMfma16x16x128_BF6FP4` has FPSan coverage without a
  matching layout test, `ScaledMfma32x32x64_FP4FP6` has FPSan coverage without
  a matching layout test, and the `Mixed8xSub` suites cover more layout
  combinations than FPSan combinations.

Closure:

- Decide whether one-sided mixed sub-byte matrix combinations are intentional
  smoke coverage or should be promoted to paired layout/FPSan tests.
- Add targeted special-value and rounding tests only where the ISA defines
  stable behavior that can be checked independently.

### Atomics have good exact-input coverage, but limited edge coverage

Atomic tests are **HIP portable** and have real behavioral oracles for final
values and `fadd` returned old values. They are still limited to
order-independent exact inputs.

Remaining limitations:

- Float fmin/fmax edge behavior for NaN and signed zero is not covered.
- Cross-architecture float atomic behavior is only validated for the current
  exact normal input domain. The ISA docs differ on denormal handling for some
  memory float atomics, especially `ADD_F32`, so these HIP-portable tests should
  not be read as proving full IEEE edge-case equivalence across RDNA4,
  gfx94x/CDNA3, and gfx950/CDNA4.
- Contended returned-old behavior is not checked because it is inherently
  schedule-dependent.
- Address-space variants are not separated; HIP selects the actual primitive
  from pointer address space.

## Coverage limitations, not immediate failures

### Generic math tests are not GPU intrinsic tests

`math_test.cpp` is **Host/generic** coverage. It covers core FPSan math
semantics, tagged-op identities, and Float-mode host `std::` parity. It is not
intended to validate AMDGPU intrinsic or emulator behavior. Do not count those
tests as behavioral coverage for `__builtin_amdgcn_*` scalar math instructions.

### FPSan-mode tests usually do not validate hardware instructions

For many wrappers, FPSan mode intentionally replaces the hardware instruction
with payload-domain algebra. These tests are necessary for FPSan correctness,
but they cannot catch an emulator bug in the hardware instruction unless the
FPSan path still executes that instruction or is compared against a Float-mode
instruction result with an independent oracle.

## Currently strong emulator-catching areas

These tests have independent references and can catch buggy hardware/emulator
behavior within their tested domains:

- **gfx950/CDNA4**: scaled conversion tests in
  `cvt_scalef32_*_gfx950_test.cpp`.
- **gfx950/CDNA4**: non-scaled fp8/bf8 all-byte unpack and exact round-trip
  tests in `cvt_gfx950_test.cpp`.
- **Shared gfx12/gfx94x/gfx950**: `amdgcn_ldexpf`, `amdgcn_ldexp`, and
  `amdgcn_ldexph`.
- **gfx12, gfx94x/CDNA3, and gfx950/CDNA4 in their respective matrix suites**:
  WMMA, MFMA, SMFMAC, and SWMMAC layout/dataflow tests on exact small inputs.
- **Shared gfx12/gfx94x/gfx950**: wave fadd/fmin/fmax reductions over the
  runtime wave size.
- **gfx12**: cross-lane readlane/readfirstlane, XOR `ds_bpermute`/`ds_permute`,
  identity DPP, and `ballot_w32` for the tested mappings.
- **HIP portable**: floating-point atomic final-value tests and the
  single-thread `atomic_fadd_f32` returned-old test.
