# QuietBox: coverage for the 640-token ISL

**Branch:** `mbezulj/refactor_prefill_tests`
**Box:** QuietBox — 4 chips (`P300_X2`). Mesh shapes it opens: `1x4` TorusX, `2x2` Fabric2D,
`4x1` TorusY.

## Read this first: there is no perf gate for QuietBox

Checked directly against the tree — **`tests/perf/` declares no `1x4`, `2x2` or `4x1` mesh anywhere.**
Every perf gate targets Galaxy `8x4`, a Galaxy `4x4` subtorus, or LoudBox `8x1`/`2x4`. So this
refactor produces **no perf re-baselining work on QuietBox**, and there is nothing here to
re-measure. That is a coverage gap in the suite, not an omission in this plan — see "Gap worth
filing" below.

The only perf test that runs on this SKU at all is `sparse_mla/test_sparse_mla_perf.py`'s QuietBox
row (`SP=1 x TP=4`, chunk=640, cache=6.25k, TorusX). It was **not touched** by this refactor —
`sparse_mla/` measures per-chip KVPE cache depth, which is a different axis from the prefill ISL — so
its baseline is still valid. Run it only as a smoke check that the box is healthy:

It takes its mesh from the physical device count (`sparse_mla_mesh.py`:
`MESH_SHAPES_BY_DEVICE_COUNT[4] = [(1,1), (1,4), (2,2)]`), so there is no box selector to pass — just
run it on the box:

```bash
pytest models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py -rs -q
```

## What QuietBox is actually for here: 4-chip functional coverage

These are migrated to 640 tokens/chip and collection-verified, but have **never executed** at the new
length. A 32-chip galaxy cannot run them — `conftest.py:285-288` skips any mesh whose device count
differs from the devices present, so every 4-chip case skips there.

| Test | QB meshes | Note |
|---|---|---|
| `pcc/test_rmsnorm.py::test_rmsnorm_distributed` | `1x4` | Only unrun half; `test_rmsnorm_single_chip` already passes at `isl_5k` |
| `pcc/test_ffn.py::test_ffn_pcc` | `1x4` | Was `[4096, 3200]`, now one 640 row |
| `pcc/test_shared_expert.py::test_shared_expert_pcc` | `1x4` | 3 rows → 2; keeps the K3 6144-intermediate case |
| `pcc/test_parallel_embedding.py::test_parallel_embedding` | `1x4` | Id was the stale `deepseek_prefill_100K` |
| `pcc/test_moe_routing_setup.py` | `4x1` | Also has LB meshes |
| `pcc/test_lm_head.py::test_lm_head` | `1x4`, `2x2` | `full-no-pcc` moved; `small` deliberately keeps 32 (see below) |
| `pcc/test_moe_gate_prefill2d.py` | `2x2` | `GATE_SP_DIM` now imported, value unchanged |
| `test_mla.py` | `2x2` | Exercises the re-pointed `scale_down_sl` at sp=2 |
| `op_unit_tests/test_zero_padded_kv_cache.py`, `test_moe_padding_config.py` | `2x2`, `4x1` | Constants-only change |

### How to run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export DEEPSEEK_V3_HF_MODEL=/mnt/models/DeepSeek-R1-0528-fp8-4236a6af/
# No MESH_DEVICE: _is_galaxy_env() must stay false so galaxy tests self-skip.

pytest models/demos/deepseek_v3_d_p/tests/pcc/ -k "torus-x-1x4 or fabric2d-mesh-2x2 or torus-y-4x1" -rs -q --tb=short
pytest models/demos/deepseek_v3_d_p/tests/test_mla.py -k "2x2" -rs -q --tb=short
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_zero_padded_kv_cache.py \
       models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_moe_padding_config.py -rs -q --tb=short
```

`-rs` matters: it prints skip reasons, which is how you tell "this box cannot form that mesh" from
"the test is broken".

## Two things not to change

- **`test_lm_head`'s `small` row keeps `batch_seq_len=32`.** Line 122 gates the PCC path on
  `batch_seq_len == ttnn.TILE_SIZE`, and 32 *is* TILE_SIZE. Moving it to 640 would skip the PCC check
  and delete that test's only correctness coverage while still reporting green.
- **`test_masked_bincount`'s `sp_dim=4096` stays.** `rows_per_core = sp_dim // 64` is 64 at 4096 and
  10 at 640; that test exists to expose a `gather_sem` race whose window scales with per-core work.
  The reason is recorded in-code.

If either looks like a missed migration, it is not — both are deliberate, per the "a row that exists
for a non-ISL reason keeps its reason" rule.

## Gap worth filing

QuietBox has **no** MoE, MLA or prefill-block device-perf coverage — `perf/` never declares a 4-chip
mesh, though `conftest.py`'s `CI_ALLOWED_FABRICS` permits `4x1` TorusY, `2x2` Fabric2D and `1x4`
TorusX on `P300_X2`. `4x1` is the natural SP-axis proxy on 4 chips and is completely unused.

Do not present a QuietBox functional pass as SKU perf coverage. If the team wants a 4-chip perf
proxy, that is new work with its own baseline, not a re-measurement of anything here.

## Acceptance

1. The `pcc/` 4-chip sweep passes, or each failure is attributed to a cause other than the ISL change.
2. `test_mla` at `2x2` passes — that is the re-pointed `scale_down_sl` at sp=2, where a direction
   error would show as 2560 tokens/chip instead of 640.
3. Skip reasons reviewed with `-rs`: every skip should name a mesh/topology constraint, not a fixture
   or import failure.
