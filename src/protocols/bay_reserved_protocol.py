"""Complete reserved-period evaluation protocol for 2026 data.

Predeclares exact cohorts, quality rules, tasks, baselines, and accounting
before reading or evaluating 2026 labels.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_METADATA = {
    "version": "2026-09-18-reserved-protocol-v1",
    "status": "FROZEN_PREDECLARED",
    "disclosure": (
        "The 2026 observation archive was previously inspected during initial "
        "forensic audits and early exploratory checks. This evaluation is an "
        "archival reserved-period chronological stress test, NOT a pristine untouched "
        "blind test. Hyperparameters and model architectures were frozen prior to this evaluation."
    ),
    "time_range": {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-06-19T23:59:59Z",
        "quarters": {
            "Q1": {"start": "2026-01-01T00:00:00Z", "end": "2026-04-01T00:00:00Z"},
            "Q2": {"start": "2026-04-01T00:00:00Z", "end": "2026-06-20T00:00:00Z"},
        }
    },
    "eligible_sensors": {
        "temperature": [
            "Nodo_P03", "Nodo_S01", "Nodo_S02", "Nodo_S03", "Nodo_S04",
            "Nodo_S05", "Nodo_S07", "Nodo_S08", "Nodo_S09",
            "TBaiaP_01m-CTD(calibrato)", "TMitiliE_05m-CTD(calibrato)", "TPortovenereP_02m-CTD(calibrato)"
        ],
        "oxygen": [
            "TBaiaP_01m-CTD(calibrato)", "TMitiliE_05m-CTD(calibrato)", "TPortovenereP_02m-CTD(calibrato)"
        ]
    },
    "quality_rules": {
        "source": "Raw Gen2 exports only; hourly labels at interval end (closed='left', label='right')",
        "bounds": {
            "temperature": [-2.0, 40.0],
            "oxygen": [0.0, 600.0],
            "salinity": [0.0, 42.0]
        },
        "sentinel_value": -999.0,
        "imputation": "None for evaluation targets or issuance anchors; missing is missing."
    },
    "tasks": {
        "known_sensor_forecast": {
            "horizon_hours": 6,
            "grid_hours": 6,
            "grid_anchors_utc": [0, 6, 12, 18],
            "anchor_requirement": "Actual valid target observation required at issuance time t.",
            "lead_window": "Target observation required at t + 6h.",
            "baselines": ["persistence"],
            "interval_policy": "rolling30",
            "nominal_coverage": 0.90,
            "interval_requirement": "Minimum 30 causal residuals available by issuance (endpoint + 1h = t + 1h)."
        },
        "temperature_outage": {
            "durations_hours": [24, 72],
            "schedule": "Weekly non-overlapping outages starting Mondays 00:00:00 UTC",
            "anchor_requirement": "Last valid real temperature in 6h strictly before outage start.",
            "policy": "DEGRADED_PERSISTENCE",
            "baselines": ["persistence"]
        },
        "spatial_field_ctd": {
            "casts_source": "data/processed/ctd_all_casts.csv.gz (qc_flag==0, 2026 casts)",
            "boundary_blocker": (
                "Copernicus Med NetCDF boundary products terminate at 2026-01-16T00:00:00Z. "
                "Casts after 2026-01-16 lack contemporaneous boundary forcing and trigger "
                "degraded_background_only fallback or spatial gate abstention."
            )
        },
        "oxygen_warning_research": {
            "status": "RESEARCH_ONLY",
            "policy": "Two consecutive hourly decline rate alerts (>1.5 umol/L/h) plus 6h cooldown.",
            "evaluation": "Recall of threshold crossings and unmatched alert burden per site-week."
        }
    },
    "accounting_denominators": [
        "scheduled_endpoints",
        "observed_targets",
        "valid_issuance_anchors",
        "supported_predictions",
        "available_intervals",
        "covered_intervals",
        "abstained_cases"
    ]
}


def export_protocol():
    path = ROOT / "config/bay_reserved_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PROTOCOL_METADATA, indent=2), encoding="utf-8")
    print(f"Predeclared reserved protocol frozen at {path}")
    return PROTOCOL_METADATA


if __name__ == "__main__":
    export_protocol()
