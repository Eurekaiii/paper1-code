"""
AeroMoDE — Complete Experiment Suite
======================================
Usage:
    python -m experiments.run_all              # run everything
    python -m experiments.run_all --quick      # quick subset
    python -m experiments.run_all --plot-only  # only regenerate figures from saved data
    python -m experiments.run_all --no-plot    # run but skip plotting
"""

from __future__ import annotations
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.scenarios import build_all_scenarios, build_sensitivity_scenarios
from experiments.runner import run_all_scenarios, save_report, load_report
from experiments.baselines import BASELINES
from experiments.plotting import generate_all_figures


# ======================================================================
# Output paths
# ======================================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

MAIN_RESULT_FILE = os.path.join(RESULTS_DIR, "main_results.json")
SENSITIVITY_PREFIX = os.path.join(RESULTS_DIR, "sensitivity")


# ======================================================================
# Main
# ======================================================================
def main():
    args = set(sys.argv[1:])

    quick = "--quick" in args
    plot_only = "--plot-only" in args
    no_plot = "--no-plot" in args

    # ---- Which baselines to run ----
    if quick:
        baselines = ["AeroMoDE", "NoSubstitute", "RandomPlacement"]
    else:
        baselines = list(BASELINES.keys())

    print("=" * 60)
    print("  AeroMoDE — Experiment Suite")
    print(f"  Baselines: {baselines}")
    print(f"  Mode: {'plot-only' if plot_only else 'quick' if quick else 'full'}")
    print("=" * 60)

    # ================================================================
    # Main experiments
    # ================================================================
    main_report = None

    if not plot_only:
        print("\n\n[Phase 1] Building main scenarios ...")
        main_scenarios = build_all_scenarios(rng_seed=42)
        print(f"  Generated {len(main_scenarios)} scenarios across "
              f"6 dimensions.")

        print("\n[Phase 2] Running main experiments ...")
        main_report = run_all_scenarios(main_scenarios, baselines, verbose=True)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_report(main_report, MAIN_RESULT_FILE)
    else:
        print("\n[Phase 1-2 skipped] Loading saved results ...")
        if os.path.exists(MAIN_RESULT_FILE):
            main_report = load_report(MAIN_RESULT_FILE)
            print(f"  Loaded {len(main_report.scenarios)} scenarios.")
        else:
            print(f"  No saved results at {MAIN_RESULT_FILE}")
            return

    # ================================================================
    # Sensitivity experiments
    # ================================================================
    sensitivity_reports = {}

    if not plot_only:
        print("\n\n[Phase 3] Building sensitivity scenarios ...")
        sens_scenarios = build_sensitivity_scenarios(rng_seed=42)
        print(f"  Parameters: {list(sens_scenarios.keys())}")

        for param, sc_list in sens_scenarios.items():
            print(f"\n  --- Sensitivity: {param} ({len(sc_list)} values) ---")
            report = run_all_scenarios(sc_list, baselines, verbose=True)
            sensitivity_reports[param] = report
            fpath = f"{SENSITIVITY_PREFIX}_{param}.json"
            save_report(report, fpath)
    else:
        print("\n[Phase 3 skipped] Loading saved sensitivity results ...")
        for param in ["xi", "lambda", "eta", "nu"]:
            fpath = f"{SENSITIVITY_PREFIX}_{param}.json"
            if os.path.exists(fpath):
                sensitivity_reports[param] = load_report(fpath)
                print(f"  Loaded sensitivity/{param} "
                      f"({len(sensitivity_reports[param].scenarios)} pts)")

    # ================================================================
    # Plotting
    # ================================================================
    if not no_plot and main_report is not None:
        print("\n\n[Phase 4] Generating figures ...")
        generate_all_figures(
            main_report,
            sensitivity_reports if sensitivity_reports else None,
            FIGURES_DIR,
        )

    print("\n\nDone. Summary:")
    if main_report:
        print(f"  Main scenarios:      {len(main_report.scenarios)}")
        for sr in main_report.scenarios[:5]:
            print(f"    {sr.scenario_name:<30s}  "
                  f"AeroMoDE={_get_ms(sr, 'AeroMoDE'):>8.1f} ms")
        if len(main_report.scenarios) > 5:
            print(f"    ... ({len(main_report.scenarios) - 5} more)")
    for param, report in sensitivity_reports.items():
        print(f"  Sensitivity {param}:     {len(report.scenarios)} points")


def _get_ms(sr, bname: str) -> float:
    """Helper: get D_total_ms for a scenario result."""
    return sr.baselines.get(bname, {}).get("D_total_ms", float("nan"))


if __name__ == "__main__":
    main()
