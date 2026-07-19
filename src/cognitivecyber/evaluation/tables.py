"""Export a results DataFrame to CSV, Excel, LaTeX, and Markdown."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table(df: pd.DataFrame, out_dir: Path, name: str, float_format: str = "%.4f"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False, float_format=float_format)
    try:
        df.to_excel(out_dir / f"{name}.xlsx", index=False)
    except ImportError:
        pass  # openpyxl not installed; CSV/Markdown/LaTeX still produced
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    latex = df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        caption=name.replace("_", " ").title(),
        label=f"tab:{name}",
    )
    (out_dir / f"{name}.tex").write_text(latex)
