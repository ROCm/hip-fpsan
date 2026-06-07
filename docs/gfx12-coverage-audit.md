# gfx12 (RDNA4) FP-intrinsic coverage audit

Authoritative inventory of every floating-point-touching
`__builtin_amdgcn_*` LLVM has tagged for gfx12 selection, with what
hip-fpsan covers in **Float** (pass-through to the builtin) and
**FPSan** (payload-ring software dataflow) modes.

Probe methodology — same shape as the gfx950 audit
(commit `4fc4632`):

1. **Enumerate.** Walk
   `BuiltinsAMDGPU.def` (ROCm 7.13 toolchain) and pick every TARGET_BUILTIN
   and BUILTIN whose signature contains an FP type letter (`f`, `d`, `h`,
   `y`). Yields **253** candidate names.

2. **Probe `__has_builtin` on a gfx1201 device compile.** Macro-expand
   each candidate inside a `--offload-arch=gfx1201` HIP source and grep
   the comments that survive. Yields **105** available on gfx1201.

3. **Audit the diff vs the wrapped surface.** Sort + comm against
   the actual builtins our headers call. Yields **55** candidates not
   yet wrapped.

4. **Isel-check the in-scope subset** (the ones that aren't excluded by
   policy). Compile a tiny kernel that calls each builtin against
   `--offload-arch=gfx1201`. All survivors lower cleanly.

## Status: completable in this scope is closed

Every FP-relevant gfx12-selectable builtin is now **wrapped in both
Float and FPSan modes**, or **deliberately deferred with rationale**.
The remaining deferrals are FPSan-unrepresentable IEEE bit-twiddling
micro-ops, graphics / ray-tracing / image-format conversions — same
policy as gfx950.

Of-course-MFMA-doesn't-exist-on-RDNA aside, gfx12 now matches gfx950
in coverage: every shape ships both modes and is silicon-verified.

## Coverage by family

### Wrapped in both modes (Float + FPSan), this turn

| Family | Wrapper | Tests on real gfx1201 |
|---|---|---|
| `global_load_tr_b128_{f16,bf16}` | `amdgcn_global_load.hpp` | 2/2 PASS — bit-faithful transpose, Float==FPSan==raw |
| `dot4_f32_{fp8,bf8}_{fp8,bf8}` (4 ops) | `amdgcn_math.hpp` | 4/4 PASS — Float-bit-exact, FPSan ring expanded |
| `swmmac_f32_16x16x32_{f16,bf16}_w32` | `amdgcn_swmmac_gfx12.hpp` | Layout & FPSan both PASS — per-lane A/B/D + sparse-index layouts reverse-engineered from the AMD matrix instruction calculator + RDNA4 ISA PDF, confirmed bit-for-bit on silicon by `LayoutMatchesHardware` / `FpsanMatchesScalarReference` |
| `swmmac_f16_16x16x32_f16_w32` | `amdgcn_swmmac_gfx12.hpp` | Layout & FPSan both PASS (same shared dataflow as f32-out, f16 accumulator) |
| `swmmac_f32_16x16x32_{fp8,bf8}_{fp8,bf8}_w32` (4 ops) | `amdgcn_swmmac_gfx12.hpp` | Layout & FPSan both PASS — distinct fp8 dataflow (different lane split, linear byte-K mapping) |

### Wrapped before this turn (background)

Already complete on gfx12 in both modes from earlier work — see
the README support matrix and the headers themselves:

- `amdgcn_wave.hpp` — wave_reduce_f32 family (4 ops),
  permlane{16,x16,64}, mov_dpp / update_dpp / mov_dpp8,
  ballot_w32, readlane / readfirstlane / writelane,
  ds_bpermute / ds_permute / ds_swizzle.
- `amdgcn_math.hpp` — rcp/rsq/sqrt/sin/cos/log/exp2/fract/tanh/fmed3
  (per-type), ldexp, fdot2 family.
- `amdgcn_atomic.hpp` — atomic_fadd_f{32,64} / fmin / fmax /
  pk_add_f16 / pk_add_bf16 (dispatched to the right address-space
  hardware op via HIP `atomicAdd`).
- `amdgcn_classify.hpp` — class / classf / classh / fcmp / fcmpf.
- `amdgcn_cvt.hpp` — cvt_pkrtz, cvt_f32_{fp8,bf8}, cvt_pk_f32_{fp8,bf8},
  cvt_pk_{fp8,bf8}_f32 (4 ops).
- `amdgcn_matrix.hpp` — gfx12 dense WMMA 16x16x16 (8 variants:
  f32/f16/bf16 acc × f16/bf16 inputs + fp8/bf8/bf8/fp8 inputs).

### Deferred: FPSan has no faithful payload-ring image

The same policy items the gfx950 audit ruled out apply here. A
symbolic FPSan payload is *not* the operand's IEEE bit pattern; any
op that reads/writes float bit fields or branches on
exponent/class/sign has no faithful payload-ring image, so shipping a
wrapper would risk a silently wrong answer.

| Op(s) | Why deferred |
|---|---|
| `div_scale{,f}` | IEEE division step: returns a possibly-2^64-rescaled operand AND a VCC flag selected by exponent ranges |
| `div_fmas{,f}` | FMA whose result is multiplied by 2^(±64) depending on a hidden carry-in (VCC) |
| `div_fixup{,f,h}` | Patches NaN/Inf/zero/sign special cases of a division using all three operands' classes |
| `frexp_mant{,f,h}` | Extracts the IEEE mantissa field (a bit query on the real float; undefined on a symbolic payload) |
| `frexp_exp{,f,h}` | Extracts the IEEE exponent as an int (same) |
| `trig_preop{,f}` | Emits a 53-bit segment of 2/π for trig argument reduction, indexed by the operand's exponent |
| `cube{id,sc,tc,ma}` | Cubemap face/coord selection (graphics) — outside FPSan's numeric scope |
| `interp_{mov,p1,p1_f16,p2,p2_f16}` | Pixel-shader attribute interpolation (graphics) |
| `image_bvh{_intersect_ray,_intersect_ray_h,_intersect_ray_l,_intersect_ray_lh,8_intersect_ray,_dual_intersect_ray}` | Ray-traversal BVH intersection (graphics / ray tracing) |
| `cvt_pknorm_i16` / `cvt_pknorm_u16` / `cvt_pk_u8_f32` / `cvt_off_f32_i4` | FP→int graphics-format packs (image / vertex pipeline) |

### Deferred: redundant with existing wrappers

| Op(s) | Why deferred |
|---|---|
| `ds_faddf` / `ds_fminf` / `ds_fmaxf` | Legacy gfx9-mnemonic with explicit `IiIiIb` ordering / scope / volatile immediates; the modern address-space-overloaded form is already wrapped via `amdgcn_atomic.hpp` (HIP `atomicAdd` dispatches to `ds_*` automatically). |
| `ds_atomic_fadd_f32` / `ds_atomic_fadd_v2f16` | Same as `flat`/`global` variants — covered by `amdgcn_atomic_fadd_f32` and `amdgcn_atomic_pk_add_f16` whose HIP `atomicAdd` lowering picks the right hardware instruction based on pointer address space. |
| `flat_atomic_fadd_v2f16` / `global_atomic_fadd_v2f16` / `global_atomic_fadd_f32` | Same. |

## Path to close the SWMMAC FPSan gap

This is the only family on gfx12 where shipping is blocked on
silicon work, not on policy. The recipe mirrors what
`include/fpsan/amdgcn_smfmac.hpp` describes for gfx950 SMFMAC:

1. **Single-hot probe for A.** Set `A_comp[i*][c*] = 1`, B = all-1
   delta, C = 0; vary i, c. For each `(i*, c*)`, the lane/reg where
   the output D ends up non-zero tells you which lane.reg in the
   per-lane storage carried `A_comp[i*][c*]`. Hypothesis to test
   first: A_comp's layout is the gfx12 dense 16x16x16 A layout
   (already nailed in `Wmma16x16x16Layout` in
   `amdgcn_matrix.hpp`), because A_comp is stored as v8 just like
   dense A.

2. **Single-hot probe for B.** Symmetric: set `B[k*][j*] = 1`, A = a
   known dense pattern, observe D. B has K=32, so the per-lane v16
   layout extends the dense K=16 B by one more K bit; figure out
   where it lives.

3. **Index decode.** Vary the 16-bit `idx` argument with a known A
   that has all-1s at every K, observe which 2 of every 4 K
   positions actually appear in the dot product. Lane→nibble mapping
   tells you which (i, group) each lane's 16-bit index covers.

4. **LayoutMatchesHardware test.** Random A/B/idx, run the builtin
   side-by-side with the proposed software dataflow, require
   bit-exact equality across the wave. Multi-seed.

5. **Refuse to ship** until 4 holds for every variant in the family.

Once that infrastructure exists, dropping the
`static_assert(always_false)` and switching to the software
dataflow is a one-line change per macro instantiation.

## Reproduce this audit

```bash
DEF=$(realpath ~/therock_venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def)

# Step 1: enumerate FP-touching builtins.
python3 -c '
import re
fp = set("fdhy")
pat_t = re.compile(r"^TARGET_BUILTIN\(__builtin_amdgcn_([A-Za-z0-9_]+),\s*\"([^\"]*)\"")
pat_b = re.compile(r"^BUILTIN\(__builtin_amdgcn_([A-Za-z0-9_]+),\s*\"([^\"]*)\"")
seen=set(); names=[]
for line in open("'"$DEF"'"):
    m = pat_t.match(line) or pat_b.match(line)
    if not m: continue
    n,s = m.group(1), m.group(2)
    if set(s) & fp and n not in seen:
        seen.add(n); names.append(n)
print("\n".join(names))
' > /tmp/fp_names.txt

# Step 2: probe __has_builtin on gfx1201.
python3 -c '
for n in open("/tmp/fp_names.txt").read().split():
    print(f"#if __has_builtin(__builtin_amdgcn_{n})\n// HAS: {n}\n#endif")
' > /tmp/probe.hip
$ROCM/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1201 -nogpulib \
    --cuda-device-only -E -C /tmp/probe.hip 2>&1 \
  | awk '/^\/\/ HAS:/{print $3}' | LC_ALL=C sort -u > /tmp/gfx1201_avail.txt

# Step 3: diff against our wrapped surface.
grep -hoE '__builtin_amdgcn_[A-Za-z0-9_]+' include/fpsan/amdgcn_*.hpp \
  | sed 's/^__builtin_amdgcn_//' | LC_ALL=C sort -u > /tmp/wrapped_sorted.txt
LC_ALL=C comm -23 /tmp/gfx1201_avail.txt /tmp/wrapped_sorted.txt > /tmp/gaps.txt
```
