# gfx94x (CDNA3 / MI300) FP-intrinsic coverage audit

Authoritative inventory of every floating-point-touching
`__builtin_amdgcn_*` LLVM exposes to a gfx942 device compile in the local
ROCm 7.14 toolchain, with what hip-fpsan covers in **Float**
(pass-through to the builtin) and **FPSan** (payload-ring software dataflow)
modes.

Probe methodology -- same shape as the gfx12 and gfx950 audits:

1. **Enumerate.** Walk the generated
   `BuiltinsAMDGPU.inc` table from the workspace ROCm 7.14 toolchain and pick
   every `Builtin::Info` whose signature contains an FP type letter. For the
   generated table this means `f`, `d`, `h`, `y`, plus `x` for the generated
   spelling of some `_Float16` signatures. Yields **499** candidate names.

2. **Probe `__has_builtin` on a gfx942 device compile.** Macro-expand each
   candidate inside a `--offload-arch=gfx942` HIP source and grep the comments
   that survive. Yields **149** available on gfx942.

3. **Audit the diff vs the wrapped surface.** Sort + comm against the actual
   builtins our headers call. Yields **82** gfx942-visible candidates already
   called by wrappers and **67** gfx942-visible candidates not called by
   wrappers.

4. **Isel-check the in-scope subset.** Compile tiny kernels for candidates
   where `__has_builtin` is known to over-report. The supported families below
   lower and pass on real gfx942 silicon. The backend-deferred families below
   are visible to `__has_builtin` but either reject with "intrinsic not
   supported on subtarget" or fail instruction selection.

GPU commands must run outside the sandbox because the sandbox cannot access ROCm
devices. Preprocess/compile probes do not need GPU access.

## Status: gfx94x support is not yet a 100% close

The CDNA3 matrix, sparse-matrix, baseline FP8/BF8 conversion, f32 scalar math,
`fdot2`, wave-f32, `ldexp`, classify/compare, and atomic families claimed below
are wrapped in both modes and silicon-verified on gfx942.

The known open implementation gap is the gfx942-visible
`xf32` MFMA pair. There is also a smaller test-coverage gap for already-wrapped
half/double scalar math wrappers that are visible on gfx942 but do not yet have a
CDNA3-specific direct-builtin-vs-wrapper test.

The last full gfx942 run for the claimed surface was clean:

```bash
ctest --test-dir build_cdna3 --output-on-failure
# 304/304 passed, 2 expected WaveFsubF32 skips
```

The same source passed after reconfiguring `build_hip` to
`CMAKE_HIP_ARCHITECTURES=gfx942`, and the pure C++ suite passed separately.

## Coverage by family

### Wrapped in both modes, this turn

| Family | Wrapper | Tests on real gfx942 |
|---|---|---|
| CDNA3 AMD-FNUZ FP8/BF8 scalar formats | `fp8.hpp`, `mix.hpp` | `fp8_test.cpp` validates the AMD FNUZ encodings: E4M3 bias 8, E5M2 bias 16, no infinity, `0x80` NaN, fixed points, exhaustive float round-trips, FPSan fixed points, and cast round-trips. |
| Baseline FP8/BF8 conversion | `amdgcn_cvt.hpp` | `cvt_test.cpp` passes on gfx942 for `cvt_f32_{fp8,bf8}`, `cvt_pk_{fp8,bf8}_f32`, `cvt_pk_f32_{fp8,bf8}`, `cvt_pkrtz`, and stochastic `cvt_sr_{fp8,bf8}_f32`; Float mode hits hardware, FPSan mode checks byte/payload plumbing. |
| Scalar f32 math | `amdgcn_math.hpp` | `amdgcn_math_cdna3_test.cpp` passes for `rcpf`, `sqrtf`, `rsqf`, `rsq_clampf`, `sinf`, `cosf`, `logf`, `exp2f`, `fractf`, and `fmed3f`; Float bit-exact against the direct builtin, FPSan payload-exact against the scalar FPSan op. |
| `fdot2` | `amdgcn_math.hpp` | `amdgcn_math_cdna3_test.cpp` passes for `fdot2` f16x2 dot into f32 accumulator; Float bit-exact against the builtin, FPSan expanded to `acc + a0*b0 + a1*b1`. |
| CDNA3 MFMA dense f16 / BF16-1K | `amdgcn_mfma.hpp` | `mfma_cdna3_test.cpp` passes layout and FPSan scalar-reference tests for f16 shapes plus BF16 `_1k` shapes. Layouts are anchored by `LayoutMatchesHardware` against the real builtin. |
| CDNA3 MFMA AMD-FNUZ FP8/BF8 | `amdgcn_mfma.hpp` | `mfma_cdna3_test.cpp` passes `16x16x32` and `32x32x16` for all `fp8/bf8` operand combinations using `amd_fp8_e4m3` / `amd_fp8_e5m2`, not OCP FP8. |
| CDNA3 MFMA f32/f64 | `amdgcn_mfma.hpp` | `mfma_cdna3_test.cpp` passes f32 `16x16x4`, `16x16x1`, `32x32x2`, `32x32x1`, `4x4x1` and f64 `16x16x4`, `4x4x4`; each has layout and FPSan scalar-reference tests. |
| CDNA3 SMFMAC f16/BF16 | `amdgcn_smfmac.hpp` | `smfmac_cdna3_test.cpp` passes `16x16x32_{f16,bf16}` and `32x32x16_{f16,bf16}`; Float checked against host sparse matmul, FPSan checked against payload reference. |
| CDNA3 SMFMAC AMD-FNUZ FP8/BF8 | `amdgcn_smfmac.hpp` | `smfmac_cdna3_test.cpp` passes `16x16x64_{fp8,bf8}_{fp8,bf8}` and `32x32x32_{fp8,bf8}_{fp8,bf8}`; all four operand combinations per shape. |

### Wrapped before this turn, now verified on gfx942

Already implemented shared wrappers also pass the gfx942 suite:

- `amdgcn_wave.hpp` -- wave_reduce_f32 family (4 ops). Two fsub
  host-reference tests are expected skips because the hardware reduction order
  is not a stable scalar fold; strategy-invariance tests pass.
- `amdgcn_math.hpp` -- `ldexpf`, `ldexp`, `ldexph`, all covered by
  `amdgcn_ldexp_test.cpp` on gfx942.
- `amdgcn_atomic.hpp` -- f32/f64 add/min/max and packed v2f16/v2bf16 add,
  covered by `atomic_test.cpp`. The wrapper intentionally uses HIP pointer
  atomics so address-space-specific ds/flat/global lowering is selected by HIP.
- `amdgcn_classify.hpp` -- `class`, `classf`, `classh`, `fcmp`, `fcmpf`, covered
  by `classify_test.cpp` on gfx942.
- Portable GPU dataflow helpers used by matrix/wave tests -- mbcnt,
  ds_bpermute, shfl/permutation helpers, and basic HIP device parity tests.

### Existing wrappers visible on gfx942, not yet separately tested here

These are called by headers and are visible to `__has_builtin` on gfx942, but the
CDNA3-specific math test currently covers only f32 scalar forms plus `fdot2`.
Do not claim full CDNA3 scalar-math closure until these have direct builtin vs
wrapper tests on real gfx942:

| Op(s) | Current status |
|---|---|
| `rcp`, `rcph` | Wrapper exists; add gfx942 Float/FPSan tests. |
| `sqrt`, `sqrth` | Wrapper exists; add gfx942 Float/FPSan tests. |
| `rsq`, `rsqh`, `rsq_clamp` | Wrapper exists; add gfx942 Float/FPSan tests. |
| `sinh`, `cosh` | Half-precision sin/cos builtins; wrapper exists; add gfx942 Float/FPSan tests. |
| `fract`, `fracth` | Wrapper exists; add gfx942 Float/FPSan tests. |
| `fmed3h` | Wrapper exists; add gfx942 Float/FPSan tests. |

`log_clampf` is different: the wrapper exists and `__has_builtin` reports it on
gfx942, but a direct gfx942 lowering probe rejects it as unsupported on the
subtarget. It is listed under backend deferrals below and should not be added to
the CDNA3 runtime tests unless the backend behavior changes.

## Deferred: FPSan has no faithful payload-ring image

The same policy items the gfx12 and gfx950 audits ruled out apply here. A
symbolic FPSan payload is not the operand's IEEE bit pattern; any op that reads
or writes float bit fields, branches on exponent/class/sign, or belongs to a
format/graphics pipeline has no faithful payload-ring image.

| Op(s) | Why deferred |
|---|---|
| `div_scale{,f}`, `div_fmas{,f}`, `div_fixup{,f,h}` | IEEE division micro-ops depend on exponent ranges, VCC, hidden carry-in, and special-case fixups. |
| `frexp_mant{,f,h}`, `frexp_exp{,f,h}` | Extract IEEE mantissa/exponent fields; undefined on symbolic FPSan payloads. |
| `trig_preop{,f}` | Argument-reduction helper indexed by the operand's exponent. |
| `cube{id,sc,tc,ma}` | Cubemap face/coord selection; graphics pipeline, not numeric FPSan scope. |
| `interp_{mov,p1,p1_f16,p2,p2_f16}` | Pixel-shader attribute interpolation; graphics pipeline. |
| `cvt_pknorm_{i16,u16}`, `cvt_pk_u8_f32`, `cvt_off_f32_i4` | FP-to-int graphics/image/vertex-format conversions. |
| `raw_buffer_{load,store}_format_v4{f16,f32}`, `struct_buffer_{load,store}_format_v4{f16,f32}` | Buffer-format memory operations with resource/format semantics, not scalar FP payload algebra. |

## Deferred: redundant with existing wrappers

These are visible to `__has_builtin`, but hip-fpsan intentionally exposes the
semantic operation instead of every address-space spelling. HIP lowers the
pointer operation to the appropriate ds/flat/global hardware instruction.

| Op(s) | Why deferred |
|---|---|
| `ds_faddf`, `ds_fminf`, `ds_fmaxf` | Legacy LDS mnemonics; covered semantically by `amdgcn_atomic_fadd_f32`, `amdgcn_atomic_fmin_f32`, and `amdgcn_atomic_fmax_f32`. |
| `ds_atomic_fadd_f{32,64}`, `flat_atomic_fadd_f{32,64}`, `global_atomic_fadd_f{32,64}` | Same fadd operation, selected by pointer address space in `amdgcn_atomic.hpp`. |
| `ds_atomic_fadd_v2f16`, `flat_atomic_fadd_v2f16`, `global_atomic_fadd_v2f16` | Covered semantically by `amdgcn_atomic_pk_add_f16`; the packed BF16 variant is also tested where the backend exposes it. |
| `flat_atomic_f{min,max}_f64`, `global_atomic_f{min,max}_f64` | Covered semantically by `amdgcn_atomic_fmin_f64` / `amdgcn_atomic_fmax_f64`; Float mode uses CAS, FPSan mode uses signed integer atomic min/max on payloads. |
| `raw_ptr_buffer_atomic_fadd_{f32,v2f16}`, `raw_ptr_buffer_atomic_f{min,max}_f64` | Buffer/raw-pointer encodings of the same atomic families; not exposed as separate FPSan APIs because hip-fpsan's public contract is pointer-based atomics, not buffer-resource descriptors. |

## Deferred: backend-visible but not usable on gfx942

These names are visible to `__has_builtin` on gfx942, but compile probes reject
them. They should not be wrapped or claimed until the backend can select them for
gfx942.

| Op(s) | Probe result |
|---|---|
| `log_clampf` | Direct call fails with `intrinsic not supported on subtarget`. |
| `wave_reduce_f{add,sub,min,max}_f64` | Direct calls fail instruction selection with 64-bit VGPR operand-class issues. |
| non-1K BF16 MFMA: `mfma_f32_16x16x2bf16`, `16x16x8bf16`, `32x32x2bf16`, `32x32x4bf16`, `4x4x2bf16` | Visible to `__has_builtin`, but direct gfx942 compile probes fail instruction selection. BF16 `_1k` forms are covered. |

## Open: in-scope gfx942 names not yet wrapped

These lower on gfx942 and are not policy-excluded. They are the current real
coverage gap relative to the gfx12/gfx950 audit standard.

| Op(s) | Required work |
|---|---|
| `mfma_f32_16x16x8_xf32`, `mfma_f32_32x32x4_xf32` | Add `amdgcn_mfma.hpp` wrappers, derive/confirm layout, add `LayoutMatchesHardware` and `FpsanMatchesScalarReference` tests on real gfx942. A tiny direct-call compile probe lowers successfully. |

## Not a CDNA3 target surface

These families are intentionally not registered for gfx94x because they are
RDNA4/gfx12 or CDNA4/gfx950 features in this project.

| Family | Reason |
|---|---|
| WMMA / SWMMAC gfx12 tests | RDNA4 wave32 matrix surface; not CDNA3. |
| `global_load_tr_b128_*` | Registered only for gfx12 in the current suite. |
| `ds_read_tr{16,8,6,4}_b*` | gfx950 matrix-transposed LDS-read wrappers/tests. |
| `cvt_scalef32_*`, fp4/fp6/BF6/MX, stochastic scaled packs | gfx950/CDNA4 scaled conversion surface. |
| scaled MFMA `mfma_scale_f32_*_f8f6f4*` | gfx950/CDNA4 scaled matrix surface. |
| SMFMAC FP8 `16x16x128` / `32x32x64` | gfx950/CDNA4 larger-K FP8 sparse shapes; CDNA3 covers `16x16x64` / `32x32x32`. |

## Path to close the gfx94x gaps

1. **Add xF32 MFMA support.** Implement wrappers for
   `mfma_f32_16x16x8_xf32` and `mfma_f32_32x32x4_xf32`; confirm the fragment
   maps with hardware layout tests before claiming FPSan correctness.

2. **Finish scalar math silicon tests.** Extend `amdgcn_math_cdna3_test.cpp` for
   the visible half/double wrappers listed above. Each test should match the
   existing pattern: direct builtin in Float mode, wrapper in Float mode,
   scalar FPSan reference, wrapper in FPSan mode.

3. **Keep backend deferrals hard-gated.** Do not add `log_clampf`, f64 wave
   reductions, or non-1K BF16 MFMA to gfx94x tests until a direct gfx942 compile
   probe lowers cleanly.

## Reproduce this audit

```bash
INC=/home/aulu/workspace/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc
CLANG=/home/aulu/workspace/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++

# Step 1: enumerate FP-touching builtins from the generated ROCm table.
python3 - <<'PY' > /tmp/fpsan_gfx94x_fp_names.txt
import re
fp = set('fdhyx')
inc = '/home/aulu/workspace/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc'
pat = re.compile(r'Builtin::Info\{Builtin::Info::StrOffsets\{\d+ /\* __builtin_amdgcn_([A-Za-z0-9_]+) \*/, \d+ /\* (.*?) \*/')
seen = set()
names = []
for line in open(inc):
    m = pat.search(line)
    if not m:
        continue
    name, sig = m.group(1), m.group(2)
    if set(sig) & fp and name not in seen:
        seen.add(name)
        names.append(name)
print('\n'.join(names))
PY

# Step 2: probe __has_builtin on gfx942.
python3 - <<'PY' > /tmp/fpsan_gfx94x_probe.hip
for n in open('/tmp/fpsan_gfx94x_fp_names.txt').read().split():
    print(f'#if __has_builtin(__builtin_amdgcn_{n})\n// HAS: {n}\n#endif')
PY
$CLANG -x hip --offload-arch=gfx942 -nogpulib \
  --cuda-device-only -E -C /tmp/fpsan_gfx94x_probe.hip 2>&1 \
  | awk '/^\/\/ HAS:/{print $3}' | LC_ALL=C sort -u \
  > /tmp/fpsan_gfx942_fp_avail.txt

# Step 3: diff against our wrapped surface.
grep -hoE '__builtin_amdgcn_[A-Za-z0-9_]+' include/fpsan/amdgcn_*.hpp \
  | sed 's/^__builtin_amdgcn_//' | LC_ALL=C sort -u \
  > /tmp/fpsan_wrapped_sorted.txt
LC_ALL=C comm -23 /tmp/fpsan_gfx942_fp_avail.txt \
  /tmp/fpsan_wrapped_sorted.txt > /tmp/fpsan_gfx942_fp_gaps.txt
LC_ALL=C comm -12 /tmp/fpsan_gfx942_fp_avail.txt \
  /tmp/fpsan_wrapped_sorted.txt > /tmp/fpsan_gfx942_fp_wrapped_avail.txt

wc -l /tmp/fpsan_gfx94x_fp_names.txt \
      /tmp/fpsan_gfx942_fp_avail.txt \
      /tmp/fpsan_gfx942_fp_wrapped_avail.txt \
      /tmp/fpsan_gfx942_fp_gaps.txt
```
