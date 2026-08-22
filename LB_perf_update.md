# LoudBox: perf re-baselining + functional coverage for the 640-token ISL

**Branch:** `mbezulj/refactor_prefill_tests`
**Box:** LoudBox — 8 chips (`P150_X8` / `T3K`). Mesh shapes it opens: `8x1` TorusY, `4x2` Fabric2D,
`2x4` Fabric2D, `1x8` TorusX.

## Why this file exists

Every test in `models/demos/deepseek_v3_d_p/tests/` now runs one input sequence length:
**640 tokens on every chip**. The global length follows the mesh (`640 * sp_factor`), so on LoudBox
it is 5120 at `8x1`, 2560 at `4x2`, 1280 at `2x4`, 640 at `1x8`.

The Galaxy 8x4 work is done (117 tests passed; the 8x4 perf gates were re-measured in
`8682d6fa2cf`). What is left needs 8 chips, and a 32-chip Blackhole cannot stand in:
`conftest.py:285-288` skips any mesh whose device count differs from the devices present, so on a
galaxy every 8-chip case skips rather than running.

## Task 1 — re-baseline three perf gates (the reason for this file)

All three are `pytest.mark.skip`-marked with `_ISL_REBASELINE_SKIP`. Their stored numbers were
measured at **3200 tokens/chip** and the rows now run **640**. Device time does not scale with ISL —
fixed dispatch overhead, CCL latency floors and expert-loop tails all stay put — so these must be
**re-measured, never rescaled**. The stored values are kept only as the reference point.

| # | Test | Mesh | Selector | Stored (stale) |
|---|---|---|---|---|
| 1a | `perf/test_moe_perf.py::test_deepseek_v3_moe_perf_loudbox` | `8x1` TorusY | `perf-host-64 and torus-y-8x1 and pad0` | `expected_ns_8x1=15_393_888` |
| 1b | same test | `2x4` Fabric2D | `perf-device-256 and fabric2d-mesh-2x4 and pad0` | `expected_ns_2x4=17_217_341` |
| 2 | `perf/test_mla_perf.py::test_deepseek_v3_mla_perf_loudbox` | `2x4` Fabric2D | `balanced and skip_check and seq5k and scaled_sl and random and fabric2d-2x4` | `expected_ns_2x4=8_857_393` |
| 3 | `perf/test_prefill_block_perf.py` id `block_2x4_layer3_moe_fabric2d` | `2x4` Fabric2D 2-link | `fabric2d-mesh-2x4-2link and layer3 and gate_device and no_ref and isl_1k28` | `38_638_478` |

**1a and 1b must be cut together.** `run_moe_perf_with_approximation` composes a single estimated 8x4
number from both CSVs — SP ops (dispatch, combine, expert FFN) from `8x1`, TP ops (gate, collectives)
from `2x4`. A stale half silently poisons the estimate, and the test will still report green.

### How to run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export DEEPSEEK_V3_HF_MODEL=/mnt/models/DeepSeek-R1-0528-fp8-4236a6af/
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden/
# No MESH_DEVICE: _is_galaxy_env() must stay false so the galaxy tests self-skip.

# Comment out the @_ISL_REBASELINE_SKIP decorator on the test you are measuring, then:
pytest models/demos/deepseek_v3_d_p/tests/perf/test_moe_perf.py::test_deepseek_v3_moe_perf_loudbox -vvv
pytest models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py::test_deepseek_v3_mla_perf_loudbox -vvv
pytest models/demos/deepseek_v3_d_p/tests/perf/test_prefill_block_perf.py -k block_2x4_layer3_moe_fabric2d -vvv
```

Each fails on the stale baseline and prints the measured value — that print is the number you want.

### Recording a new baseline

Take **at least three runs** and record the spread; the margin has to cover observed noise rather
than a guess. Stamp provenance the way `perf/test_kimi_k3_moe_perf.py:59-65` already does — box,
date, DDR speed, power, and the peak-to-peak spread that justifies the margin. Then delete the
`@_ISL_REBASELINE_SKIP` decorator (and `_ISL_REBASELINE_SKIP` itself once no test references it).

A number without provenance cannot be re-cut by anyone else, which is how baselines become permanent.

## Task 2 — functional coverage that only 8 chips can execute

None of these has ever run at 640/chip. Migrated and collection-verified, but **not executed**.

| Test | LB meshes | Priority |
|---|---|---|
| `op_unit_tests/test_ttnn_dispatch_combine.py::test_ttnn_dispatch_combine_top4` | `8x1` | **Highest** — see below |
| `pcc/test_ttnn_moe.py::test_ds_moe`, `::test_kimi_moe` | `8x1`, `4x2`, `2x4` | High — 12 rows moved here |
| `pcc/test_moe_routing_setup.py` | `8x1`, `4x2`, `2x4` | High |
| `pcc/test_moe_gate_prefill2d.py` | `8x1`, `4x2`, `2x4` | Medium |
| `test_prefill_block.py`, `test_prefill_block_loop.py`, `test_prefill_transformer.py` | `4x2`, `2x4` | Medium — `4x2`/`2x4` newly reachable, see R21 note |
| `pcc/test_shared_expert.py`, `pcc/test_parallel_embedding.py`, `pcc/test_lm_head.py` | `2x4` | Medium |
| `test_mla.py`, `test_kv_cache_table.py` | `2x4` | Medium |
| `op_unit_tests/test_zero_padded_kv_cache.py`, `test_moe_padding_config.py` | `2x4`, `4x2`, `8x1` | Low — constants-only change |

**`test_ttnn_dispatch_combine_top4` is the priority.** It carried the one genuine ISL miss the
refactor found — it was still passing `seq_len_per_chip=1600` — and its mesh is `8x1`-only, so the fix
has never executed anywhere. Treat it as the gate on merging.

```bash
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ttnn_dispatch_combine.py::test_ttnn_dispatch_combine_top4 -vvv
pytest models/demos/deepseek_v3_d_p/tests/pcc/ -k "torus-y-8x1 or fabric2d-mesh-4x2 or fabric2d-mesh-2x4" -vvv --tb=short
```

**`test_prefill_block` on `4x2`/`2x4` is new coverage, not a regression check.** An L1 guard used to
skip those meshes because 25k put 12800 tokens on each chip; at 640 it is unreachable and was removed
(R21). If those cases fail, suspect the widened matrix rather than the ISL change, and report the
per-chip shapes.

## Notes

- Run `pytest models/demos/deepseek_v3_d_p/tests/test_fabric_availability.py -vvv` first. It reports
  which fabrics the host can actually open and separates a cabling limitation from a broken test.
  It skips on a non-32-device host, so on LoudBox it is informational only.
- One production file changed: `tt/moe/tt_moe_gate_prefill.py`'s `GATE_PRODUCTION_SP_DIM` now imports
  from `utils/chunk_config.py`. The value is identical (640) — re-sourced, not retuned.
- Three `*-4x4` rows in `test_prefill_block_perf.py` carry `_SUBTORUS_4X4_HOSTGATE_SKIP`. That is a
  pre-existing host-gate issue on a **galaxy subtorus**, unrelated to the ISL work and not LoudBox work.

## Acceptance

1. Three perf gates re-measured with provenance and spread; `_ISL_REBASELINE_SKIP` removed.
2. `test_ttnn_dispatch_combine_top4` passes on `8x1`.
3. The `pcc/` LB sweep passes, or each failure is attributed to a cause other than the ISL change.
