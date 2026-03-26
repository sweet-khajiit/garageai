"""
Collect NHTSA complaints and recalls for the 2018 Audi A4.
Run this locally — requires internet access to api.nhtsa.gov.

Usage:
    python scripts/collect_nhtsa.py
"""

import json
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAKE = "audi"
MODEL = "a4"
YEAR = 2018


def collect_complaints():
    """Pull all NHTSA complaints for the 2018 Audi A4."""
    url = f"https://api.nhtsa.gov/complaints/complaintsByVehicle?make={MAKE}&model={MODEL}&modelYear={YEAR}"
    print(f"Fetching complaints from: {url}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    complaints = data.get("results") or data.get("Results", [])
    if isinstance(data, list):
        complaints = data
    print(f"Found {len(complaints)} complaints")

    # Save raw JSON
    out_path = DATA_DIR / "nhtsa_complaints.json"
    with open(out_path, "w") as f:
        json.dump(complaints, f, indent=2)
    print(f"Saved to {out_path}")

    # Also save as readable text documents for RAG ingestion
    docs = []
    for c in complaints:
        component = c.get("components") or c.get("Components", "Unknown")
        summary = c.get("summary") or c.get("Summary", "No summary")
        date = c.get("dateComplaintFiled") or c.get("DateComplaintFiled", "Unknown date")
        crash = c.get("crash") or c.get("Crash", "No")
        fire = c.get("fire") or c.get("Fire", "No")
        injuries = c.get("numberOfInjuries") or c.get("NumberOfInjuries", 0)
        odiNumber = c.get("odiNumber") or c.get("ODINumber", "N/A")

        doc = (
            f"NHTSA Complaint #{odiNumber}\n"
            f"Vehicle: {YEAR} Audi A4\n"
            f"Date Filed: {date}\n"
            f"Component: {component}\n"
            f"Crash: {crash} | Fire: {fire} | Injuries: {injuries}\n"
            f"Description: {summary}\n"
        )
        docs.append(doc)

    text_path = DATA_DIR / "nhtsa_complaints.txt"
    with open(text_path, "w") as f:
        f.write("\n---\n\n".join(docs))
    print(f"Saved readable version to {text_path}")

    return complaints


def collect_recalls():
    """Pull all NHTSA recalls for the 2018 Audi A4."""
    url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={MAKE}&model={MODEL}&modelYear={YEAR}"
    print(f"\nFetching recalls from: {url}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recalls = data.get("results") or data.get("Results", [])
    if isinstance(data, list):
        recalls = data
    print(f"Found {len(recalls)} recalls")

    # Save raw JSON
    out_path = DATA_DIR / "nhtsa_recalls.json"
    with open(out_path, "w") as f:
        json.dump(recalls, f, indent=2)
    print(f"Saved to {out_path}")

    # Save as readable text
    docs = []
    for r in recalls:
        campaign = r.get("nhtsaCampaignNumber") or r.get("NHTSACampaignNumber", "N/A")
        component = r.get("component") or r.get("Component", "Unknown")
        summary = r.get("summary") or r.get("Summary", "No summary")
        consequence = r.get("consequence") or r.get("Consequence", "Unknown")
        remedy = r.get("remedy") or r.get("Remedy", "Unknown")
        report_date = r.get("reportReceivedDate") or r.get("ReportReceivedDate", "Unknown")

        doc = (
            f"NHTSA Recall Campaign #{campaign}\n"
            f"Vehicle: {YEAR} Audi A4\n"
            f"Date: {report_date}\n"
            f"Component: {component}\n"
            f"Summary: {summary}\n"
            f"Consequence: {consequence}\n"
            f"Remedy: {remedy}\n"
        )
        docs.append(doc)

    text_path = DATA_DIR / "nhtsa_recalls.txt"
    with open(text_path, "w") as f:
        f.write("\n---\n\n".join(docs))
    print(f"Saved readable version to {text_path}")

    return recalls


if __name__ == "__main__":
    print("=" * 60)
    print("GarageAI — NHTSA Data Collection")
    print("=" * 60)
    collect_complaints()
    collect_recalls()
    print("\nDone! Check data/raw/ for output files.")
