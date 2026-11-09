# Results directory

This directory contains the raw data and derived tables from the final 720-run benchmark.
All timings are in milliseconds and were measured with NPU events around the core
`generic_block_sparse_attention` operator only.

## Files

| File | Rows | Description |
|---|---:|---|
| `raw_runs.csv` | 720 | One row per run: configuration, selection metrics, latency statistics, output sanity flag |
| `raw_timings.csv` | 36,000 | Every timed iteration (50 per run) |
| `metadata_bench.csv` | 20 | Separate metadata-operator timings, 4 repeats per batch size |
| `config.json` | - | Exact benchmark configuration and CLI values |
| `environment.json` | - | Host/container/OS/Python/CANN/torch-npu/package versions and `npu-smi info` |
| `manifest.json` | - | Completion time, run counts, SHA-256 of the two raw CSVs |
| `run_full.log` | - | Console log from the full run |
| `summary_by_scene_range_batch.csv` | 60 | Aggregated medians, IQR, page footprint, and 16k/1k latency ratio |
| `contrast_w1k_to_w16k.csv` | 15 | Direct 1k -> 16k contrast for batch 1/8/32/128/256 |
| `correlations_by_batch.csv` | 20 | Spearman correlations per batch and per scene subset |

## Key raw columns

| Column | Meaning |
|---|---|
| `scene` | `recent_contiguous`, `multi_sink`, or `uniform` |
| `range_regions_requested` | Candidate window size in logical 8-token regions: 1024/4096/8192/16384 |
| `candidate_regions_effective` | Actual usable historical regions; equals 16383 for the 16k request |
| `selected_span_mean_regions` | Mean `max(id)-min(id)+1` over all `(group, request)` selections |
| `touched_pages_mean` | Mean number of unique physical pages touched by one request, averaged over groups/requests |
| `touched_pages_union` | Number of unique physical pages touched by the entire batch |
| `kv_working_set_mib` | `touched_pages_union * page_size * num_kv_groups * head_dim * 2 * 2 / 2**20` |
| `latency_ms_median` | Median of the 50 NPU-event samples in this run |
| `latency_ms_p90` | P90 of the 50 NPU-event samples in this run |
| `out_finite` | Whether all output elements passed `isfinite`; must be `True` |
| `seed` | Selection seed (7000..7011 for the final run) |
| `kv_seed` / `block_table_seed` | Fixed KV and block-table seeds (20261105 / 20261106) |

## Aggregation rule

For every `(batch_size, scene, range_regions_requested)` cell, the report uses:

1. median of the 50 timed iterations within each run;
2. median of the 12 run medians across selection seeds;
3. IQR across the 12 run medians as the uncertainty band.

The `speedup_x_vs_w1k` column is the latency ratio `median(W=16k) / median(W=1k)` (the legacy column name says speedup). Values greater than
1 mean the wider candidate range is slower.

## Checksums

`../CHECKSUMS.sha256` covers the report, workbook, figures, scripts, and result files.
Verify from the package root with:

```bash
sha256sum -c CHECKSUMS.sha256
```