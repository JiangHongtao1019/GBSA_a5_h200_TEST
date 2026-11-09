# -*- coding: utf-8 -*-
import pandas as pd

pd.set_option("display.width", 220)
d = pd.read_csv("results/summary_by_scene_range_batch.csv")
b = d[d.batch_size == 128]
print("== B=128 ==")
print(b[["scene", "range_regions_requested", "latency_median_ms", "span_median_regions",
         "per_request_pages_median", "union_pages_median", "working_set_mib_median",
         "delta_pct_vs_w1k", "speedup_x_vs_w1k"]].round(4).to_string(index=False))

print("\n== batch deltas (1k -> 16k) ==")
for bs in sorted(d.batch_size.unique()):
    s = d[d.batch_size == bs]
    for sc in ["recent_contiguous", "multi_sink", "uniform"]:
        t = s[s.scene == sc].set_index("range_regions_requested")["latency_median_ms"]
        a, z = t.loc[1024], t.loc[16384]
        print(f"B={bs:<4} {sc:18s} {a:.4f} -> {z:.4f}  {100*(z/a-1):+6.1f}%")

print("\n== controlled contrasts ==")
s = d[d.batch_size == 128]
g = lambda sc, rg: s[(s.scene == sc) & (s.range_regions_requested == rg)].iloc[0]
a, z = g("multi_sink", 16384), g("uniform", 16384)
print(f"same ~16k span : multi {a.latency_median_ms:.4f} (span {a.span_median_regions:.0f}, per-req {a.per_request_pages_median:.1f}, union {a.union_pages_median:.0f})"
      f" vs uniform {z.latency_median_ms:.4f} (span {z.span_median_regions:.0f}, per-req {z.per_request_pages_median:.1f}, union {z.union_pages_median:.0f})"
      f" -> {z.latency_median_ms/a.latency_median_ms:.2f}x")
a, z = g("recent_contiguous", 8192), g("uniform", 8192)
print(f"same 8k range : recent {a.latency_median_ms:.4f} (span {a.span_median_regions:.0f}, per-req {a.per_request_pages_median:.1f}, union {a.union_pages_median:.0f})"
      f" vs uniform {z.latency_median_ms:.4f} (span {z.span_median_regions:.0f}, per-req {z.per_request_pages_median:.1f}, union {z.union_pages_median:.0f})"
      f" -> {z.latency_median_ms/a.latency_median_ms:.2f}x")
a, z = g("recent_contiguous", 1024), g("uniform", 1024)
print(f"same 1k range : recent {a.latency_median_ms:.4f} vs uniform {z.latency_median_ms:.4f} -> {100*(z.latency_median_ms/a.latency_median_ms-1):+.1f}%")

print("\n== correlations (all scenes) ==")
c = pd.read_csv("results/correlations_by_batch.csv")
c = c[c.subset == "all_scenes"]
print(c[["batch_size", "spearman_span_vs_latency", "spearman_per_request_pages_vs_latency",
         "spearman_union_pages_vs_latency"]].round(3).to_string(index=False))

print("\n== metadata overhead ==")
try:
    m = pd.read_csv("results/metadata_bench.csv")
    print(m.head(12).to_string(index=False))
except Exception as e:
    print("n/a", e)
