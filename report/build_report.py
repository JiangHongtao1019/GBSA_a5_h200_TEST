#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build all report artifacts for the GBSA range benchmark.

Usage:
    python report/build_report.py --data results --out .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_common import DATA, OUT, load_data, make_summaries
from report_charts import make_figures
from report_excel import make_excel
from report_markdown import make_markdown
from fig_scene_patterns import make_scene_pattern_figure


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    d = load_data(data_dir)
    raw, timings, meta = d["raw"], d["timings"], d["meta"]
    summary, contrast, corr = make_summaries(raw)

    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)
    summary.to_csv(results_dir / "summary_by_scene_range_batch.csv", index=False)
    contrast.to_csv(results_dir / "contrast_w1k_to_w16k.csv", index=False)
    corr.to_csv(results_dir / "correlations_by_batch.csv", index=False)

    figs = make_figures(summary, raw, out_dir / "figures")
    figs["fig7"] = make_scene_pattern_figure(out_dir / "figures")
    (out_dir / "figures" / "figures_manifest.json").write_text(
        json.dumps({k: str(v.relative_to(out_dir)) for k, v in figs.items()}, indent=2),
        encoding="utf-8")

    make_excel(summary, contrast, corr, raw, timings, meta, d["config"], d["env"],
               d["manifest"], figs, out_dir / "GBSA_128K_TOP512_Range_Report.xlsx")
    make_markdown(summary, contrast, corr, raw, meta, d["config"], d["env"],
                  d["manifest"], figs, out_dir / "GBSA_128K_TOP512_Range_Report.md")
    print(f"wrote report -> {out_dir / 'GBSA_128K_TOP512_Range_Report.md'}")
    print(f"wrote excel  -> {out_dir / 'GBSA_128K_TOP512_Range_Report.xlsx'}")
    print("figures:", ", ".join(p.name for p in figs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())