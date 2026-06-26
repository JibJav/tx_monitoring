import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

# ==========================================================
# Paths
# ==========================================================
REFERENCE_DATA = "data/processed/transactions_processed.csv"

CURRENT_DATA = "data/live/current.csv"

OUTPUT_DIR = "results"

OUTPUT_HTML = f"{OUTPUT_DIR}/drift_report.html"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================
print("Loading datasets...")

reference = pd.read_csv(REFERENCE_DATA)

current = pd.read_csv(CURRENT_DATA)

print("Reference rows:", len(reference))
print("Current rows:", len(current))

# ==========================================================
# Create Evidently report
# ==========================================================
report = Report(

    metrics=[

        DataDriftPreset()

    ]

)

print("Running drift analysis...")

snapshot = report.run(
    reference_data=reference,
    current_data=current
)

print("Saving HTML report...")

snapshot.save_html(
    OUTPUT_HTML
)

print("Done!")
print(f"Report saved to: {OUTPUT_HTML}")