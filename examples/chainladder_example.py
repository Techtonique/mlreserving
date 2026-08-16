"""
Side-by-side comparison: chainladder (classical) vs. MLReserving (ML + conformal PIs)

Both models are fit directly on the SAME chainladder.Triangle object -- no
manual conversion needed, thanks to MLReserving's chainladder interop:

    * MLReserving().fit(triangle)         accepts a chainladder.Triangle
    * model.to_triangle('mean')           returns a chainladder.Triangle back

This makes the two libraries drop-in comparable on the same data structure.

Run with:
    python examples/chainladder_side_by_side.py
"""
from __future__ import annotations

import warnings

import chainladder as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlreserving import MLReserving

warnings.filterwarnings("ignore")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data -- a single chainladder Triangle, used by BOTH models
    # ------------------------------------------------------------------
    triangle = cl.load_sample("raa")
    print("Input triangle (chainladder.Triangle):")
    print(triangle)
    print()

    # ------------------------------------------------------------------
    # 2. Classical chain-ladder (chainladder package)
    # ------------------------------------------------------------------
    cl_model = cl.Chainladder().fit(triangle)

    cl_ultimate = cl_model.ultimate_.to_frame(origin_as_datetime=False).iloc[:, 0]
    cl_ibnr = cl_model.ibnr_.to_frame(origin_as_datetime=False).iloc[:, 0].fillna(0.0)

    # chainladder uses a PeriodIndex for origin; MLReserving uses plain int.
    # Align them so the two series/frames below line up by origin year.
    cl_ultimate.index = cl_ultimate.index.astype(str).astype(int)
    cl_ibnr.index = cl_ibnr.index.astype(str).astype(int)

    # ------------------------------------------------------------------
    # 3. MLReserving -- fit directly on the same Triangle object
    # ------------------------------------------------------------------
    ml_model = MLReserving(
        type_pi="bootstrap",      # simulate prediction intervals via bootstrap
        replications=500,
        random_state=42,
    )
    ml_model.fit(triangle)        # <-- chainladder.Triangle accepted directly
    ml_model.predict()

    ml_ultimate = ml_model.get_ultimate()   # DescribeResult(mean, lower, upper)
    ml_ibnr = ml_model.get_ibnr()

    # ------------------------------------------------------------------
    # 4. Side-by-side comparison table
    # ------------------------------------------------------------------
    comparison = pd.DataFrame(
        {
            "chainladder_ultimate": cl_ultimate,
            "mlreserving_ultimate_mean": ml_ultimate.mean,
            "mlreserving_ultimate_lower": ml_ultimate.lower,
            "mlreserving_ultimate_upper": ml_ultimate.upper,
            "chainladder_ibnr": cl_ibnr,
            "mlreserving_ibnr_mean": ml_ibnr.mean,
        }
    ).round(0)
    comparison.index.name = "origin"

    print("Per-origin comparison:")
    print(comparison.to_string())
    print()

    totals = comparison.sum(numeric_only=True).round(0)
    print("Totals:")
    print(totals.to_string())
    print()

    pct_diff = (
        (totals["mlreserving_ultimate_mean"] - totals["chainladder_ultimate"])
        / totals["chainladder_ultimate"]
        * 100
    )
    print(f"MLReserving vs chainladder total ultimate: {pct_diff:+.1f}%")
    print()

    # ------------------------------------------------------------------
    # 5. MLReserving's completed triangle, converted BACK to chainladder
    #    -- demonstrates the round trip / full interoperability.
    # ------------------------------------------------------------------
    ml_full_triangle = ml_model.to_triangle("mean")
    print("MLReserving's completed square, as a chainladder.Triangle:")
    print(ml_full_triangle)
    print()

    # ------------------------------------------------------------------
    # 6. Chart: ultimate by origin, both methods, with MLReserving's
    #    conformal interval as error bars
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison))
    width = 0.35

    ax.bar(x - width / 2, comparison["chainladder_ultimate"], width,
           label="chainladder (classical)", color="#4C72B0")

    yerr = np.vstack([
        comparison["mlreserving_ultimate_mean"] - comparison["mlreserving_ultimate_lower"],
        comparison["mlreserving_ultimate_upper"] - comparison["mlreserving_ultimate_mean"],
    ]).clip(min=0)
    ax.bar(x + width / 2, comparison["mlreserving_ultimate_mean"], width,
           yerr=yerr, capsize=3, label="MLReserving (mean \u00b1 95% PI)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index, rotation=45)
    ax.set_ylabel("Ultimate loss")
    ax.set_title("Ultimate reserves by origin: chainladder vs. MLReserving (RAA triangle)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("chainladder_vs_mlreserving.png", dpi=150)
    print("Saved chart to chainladder_vs_mlreserving.png")


if __name__ == "__main__":
    main()