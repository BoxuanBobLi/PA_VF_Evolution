#!/usr/bin/env python3

import os
import re
import glob
import numpy as np
import pandas as pd

BASE_DIR = (
    "/data2/B_Li/vfdb/"
    "workflow_clade_translatorX_solved_dup_solved/dnds_output"
)

REPRESENTATIVE_TABLE = (
    "/data2/B_Li/vfdb/"
    "workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/output/"
    "core_VF_nonredundant_representatives.csv"
)

OUTPUT_DIR = (
    "/data2/B_Li/vfdb/"
    "workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/GO/study"
)

CONDITIONS = {
    "0_vs_0": "BB",
    "0_vs_1": "BA",
    "1_vs_0": "AB",
    "1_vs_1": "AA",
}

TOP_N = 15
VFG_RE = re.compile(r"(VFG\d+)", re.I)


def extract_vfg(text):
    match = VFG_RE.search(str(text))
    return match.group(1).upper() if match else None


def read_mean_dnds(path):
    df = pd.read_csv(path, sep=None, engine="python")
    columns = {str(c).strip().lower(): c for c in df.columns}

    col = (
        columns.get("dn/ds")
        or columns.get("dnds")
        or columns.get("omega")
        or df.columns[-1]
    )

    values = pd.to_numeric(df[col], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values >= 0]

    return values.mean() if not values.empty else np.nan


def load_condition(folder, representative_ids):
    rows = []

    for path in glob.glob(os.path.join(folder, "*.csv")):
        if not path.endswith("_groupwise_dnds.csv"):
            continue

        vfg = extract_vfg(os.path.basename(path))

        if vfg not in representative_ids:
            continue

        rows.append({
            "VFG_ID": vfg,
            "mean_dnds": read_mean_dnds(path),
        })

    return (
        pd.DataFrame(rows)
        .groupby("VFG_ID", as_index=False)["mean_dnds"]
        .mean()
    )


def create_study(df_x, df_y, x_name, y_name, output_name):
    merged = df_x.merge(
        df_y,
        on="VFG_ID",
        suffixes=("_x", "_y"),
    ).dropna()

    merged[x_name] = merged["mean_dnds_x"]
    merged[y_name] = merged["mean_dnds_y"]

    merged["y_minus_x"] = merged[y_name] - merged[x_name]
    merged["x_minus_y"] = merged[x_name] - merged[y_name]

    top_above = merged.nlargest(TOP_N, "y_minus_x")
    top_below = merged.nlargest(TOP_N, "x_minus_y")

    over_one = merged[
        (merged[x_name] > 1) |
        (merged[y_name] > 1)
    ]

    selected = (
        pd.concat([top_above, top_below, over_one])
        .drop_duplicates("VFG_ID")
        .sort_values("VFG_ID")
    )

    selected.to_csv(
        os.path.join(OUTPUT_DIR, f"{output_name}.csv"),
        index=False,
    )

    selected["VFG_ID"].to_csv(
        os.path.join(OUTPUT_DIR, f"{output_name}.txt"),
        index=False,
        header=False,
    )

    print(
        f"{output_name}: {len(selected)} unique genes "
        f"({len(top_above)} above, {len(top_below)} below, "
        f"{len(over_one)} with dN/dS > 1)"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reps = pd.read_csv(REPRESENTATIVE_TABLE)

    if "VFG_ID" in reps.columns:
        representative_ids = set(
            reps["VFG_ID"].apply(extract_vfg).dropna()
        )
    else:
        representative_ids = set(
            reps["Accession"].apply(extract_vfg).dropna()
        )

    data = {}

    for folder, label in CONDITIONS.items():
        data[label] = load_condition(
            os.path.join(BASE_DIR, folder),
            representative_ids,
        )

        print(f"{label}: {len(data[label])} representative VFs")

    # Clade A scatter: x = AA, y = AB
    create_study(
        data["AA"],
        data["AB"],
        x_name="AA",
        y_name="AB",
        output_name="study_AA_vs_AB",
    )

    # Clade B scatter: x = BB, y = BA
    create_study(
        data["BB"],
        data["BA"],
        x_name="BB",
        y_name="BA",
        output_name="study_BB_vs_BA",
    )


if __name__ == "__main__":
    main()