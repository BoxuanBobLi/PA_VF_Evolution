#!/usr/bin/env python3

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# ============================================================
# DEFAULT PATHS
# ============================================================

# Use the annotated CSV generated from VFDB_setB_pro.fas.gz.
DEFAULT_BACKGROUND = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/VF_category/"
    "core_VF_nonredundant_representatives_with_VF_category.csv"
)

DEFAULT_STUDY = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/GO/study/study_AA_vs_AB.txt"
)

DEFAULT_OUTPUT = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/VF_category/study_AA_vs_AB"
)

VFG_RE = re.compile(r"(VFG\d+)", re.IGNORECASE)
LOCUS_RE = re.compile(r"\bPA\d{4}\b", re.IGNORECASE)
MULTI_VALUE_RE = re.compile(r"\s*[;|]\s*")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Perform VFDB-category over-representation analysis using an "
            "annotated nonredundant VF CSV as background and a selected VF "
            "list as foreground."
        )
    )

    parser.add_argument(
        "--background",
        default=DEFAULT_BACKGROUND,
        help=(
            "Annotated nonredundant VF CSV containing VFG_ID, VF_category, "
            "and preferably VFC_ID and locus_tag. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--study",
        default=DEFAULT_STUDY,
        help=(
            "Foreground list containing one VFG ID or PAO1 locus tag per line. "
            "CSV/TSV files containing VFG_ID, Accession, or locus_tag also work. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory. Default: %(default)s",
    )

    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR threshold. Default: %(default)s",
    )

    parser.add_argument(
        "--min-category-count",
        type=int,
        default=2,
        help=(
            "Minimum number of annotated foreground VFs assigned to a category "
            "for that category to be tested. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--top-categories",
        type=int,
        default=20,
        help="Maximum number of categories shown in the plot. Default: %(default)s",
    )

    parser.add_argument(
        "--include-unmatched-status",
        action="store_true",
        help=(
            "Include rows with a non-'matched' VFDB_annotation_status when they "
            "still contain a VF_category. By default, only status='matched' "
            "or rows with missing status are accepted."
        ),
    )

    return parser.parse_args()


# ============================================================
# INPUT HELPERS
# ============================================================

def extract_vfg(value):
    if pd.isna(value):
        return None

    match = VFG_RE.search(str(value))
    return match.group(1).upper() if match else None


def normalize_locus_tag(value):
    if pd.isna(value):
        return None

    match = LOCUS_RE.search(str(value))
    return match.group(0).upper() if match else None


def read_table(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".tsv":
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def read_study_list(path):
    """
    Read either:
      - plain text with one ID per line;
      - CSV/TSV containing VFG_ID, Accession, or locus_tag.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension in {".csv", ".tsv"}:
        table = read_table(path)

        preferred_columns = [
            "VFG_ID",
            "Accession",
            "locus_tag",
        ]

        selected_column = next(
            (
                column
                for column in preferred_columns
                if column in table.columns
            ),
            table.columns[0],
        )

        return (
            table[selected_column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    with open(path, "r") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        ]


def split_values(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "na",
        "n/a",
        "-",
    }:
        return []

    return [
        item.strip()
        for item in MULTI_VALUE_RE.split(text)
        if item.strip()
    ]


# ============================================================
# BACKGROUND AND CATEGORY MAPPING
# ============================================================

def prepare_background(background_path):
    background = read_table(background_path)

    required = {
        "VFG_ID",
        "VF_category",
    }

    missing = required - set(background.columns)

    if missing:
        raise ValueError(
            "Annotated background CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    background["VFG_ID"] = (
        background["VFG_ID"]
        .apply(extract_vfg)
    )

    if "locus_tag" in background.columns:
        background["locus_tag"] = (
            background["locus_tag"]
            .apply(normalize_locus_tag)
        )
    else:
        background["locus_tag"] = None

    background = background[
        background["VFG_ID"].notna()
    ].copy()

    duplicate_ids = (
        background["VFG_ID"]
        .value_counts()
    )

    duplicate_ids = set(
        duplicate_ids[
            duplicate_ids > 1
        ].index
    )

    if duplicate_ids:
        print(
            f"Warning: {len(duplicate_ids)} duplicated VFG IDs were found "
            "in the background. Keeping the first row per VFG ID."
        )

    # One retained nonredundant VFG ID is one enrichment unit.
    background = background.drop_duplicates(
        subset=["VFG_ID"],
        keep="first",
    )

    return background


def build_category_mapping(
    background,
    include_unmatched_status=False,
):
    """
    Build one row per VFG_ID-category assignment from columns already present
    in the annotated background CSV.

    Expected example:
        VFG_ID       VF_category  VFC_ID
        VFG015856    Exotoxin     VFC0235
    """
    annotation = background.copy()

    if (
        "VFDB_annotation_status" in annotation.columns
        and not include_unmatched_status
    ):
        status = (
            annotation["VFDB_annotation_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        accepted_status = (
            status.eq("")
            | status.eq("matched")
        )

        annotation = annotation[
            accepted_status
        ].copy()

    rows = []

    for _, row in annotation.iterrows():
        vfg_id = row["VFG_ID"]
        categories = split_values(
            row["VF_category"]
        )

        vfc_ids = (
            split_values(row["VFC_ID"])
            if "VFC_ID" in annotation.columns
            else []
        )

        vf_names = (
            split_values(row["VF_name"])
            if "VF_name" in annotation.columns
            else []
        )

        vf_ids = (
            split_values(row["VF_ID"])
            if "VF_ID" in annotation.columns
            else []
        )

        if not categories:
            continue

        # Usually each retained VFG has exactly one category. If multiple
        # semicolon-delimited values exist, align by position when possible.
        for index, category in enumerate(categories):
            rows.append({
                "VFG_ID": vfg_id,
                "VF_category": category,
                "VFC_ID": (
                    vfc_ids[index]
                    if index < len(vfc_ids)
                    else ";".join(vfc_ids)
                    if vfc_ids
                    else None
                ),
                "VF_name": (
                    vf_names[index]
                    if index < len(vf_names)
                    else ";".join(vf_names)
                    if vf_names
                    else None
                ),
                "VF_ID": (
                    vf_ids[index]
                    if index < len(vf_ids)
                    else ";".join(vf_ids)
                    if vf_ids
                    else None
                ),
            })

    mapping = pd.DataFrame(rows)

    if mapping.empty:
        raise RuntimeError(
            "No valid VF_category values were found in the annotated "
            "background CSV."
        )

    mapping = mapping.drop_duplicates(
        subset=[
            "VFG_ID",
            "VF_category",
            "VFC_ID",
        ]
    )

    return mapping


# ============================================================
# FOREGROUND MATCHING
# ============================================================

def match_study_to_background(
    background,
    study_values,
):
    study_vfgs = {
        extract_vfg(value)
        for value in study_values
        if extract_vfg(value)
    }

    study_loci = {
        normalize_locus_tag(value)
        for value in study_values
        if normalize_locus_tag(value)
    }

    study = background[
        background["VFG_ID"].isin(study_vfgs)
        | background["locus_tag"].isin(study_loci)
    ].copy()

    matched_vfgs = set(
        study["VFG_ID"]
    )

    matched_loci = set(
        study["locus_tag"]
        .dropna()
    )

    unmatched = []

    for value in study_values:
        vfg = extract_vfg(value)
        locus = normalize_locus_tag(value)

        if (
            (vfg is None or vfg not in matched_vfgs)
            and (locus is None or locus not in matched_loci)
        ):
            unmatched.append(value)

    return study, unmatched


# ============================================================
# ENRICHMENT
# ============================================================

def perform_enrichment(
    annotated_background,
    annotated_study,
    category_mapping,
    fdr_threshold,
    min_study_count,
):
    """
    One-sided Fisher exact over-representation analysis.

    Only VFs with a valid category annotation are included in the statistical
    universe. Unannotated VFs are reported separately and are not treated as
    category-negative genes.

    For each category:

                              In category    Not in category
        Annotated study            a                b
        Annotated background
          minus study              c                d
    """
    background_vfgs = set(
        annotated_background["VFG_ID"]
        .dropna()
        .unique()
    )

    study_vfgs = set(
        annotated_study["VFG_ID"]
        .dropna()
        .unique()
    )

    study_vfgs &= background_vfgs

    background_size = len(background_vfgs)
    study_size = len(study_vfgs)

    if background_size == 0:
        raise RuntimeError(
            "No category-annotated background VFs are available."
        )

    if study_size == 0:
        raise RuntimeError(
            "No category-annotated foreground VFs are available."
        )

    results = []

    for category, group in category_mapping.groupby(
        "VF_category",
        dropna=False,
    ):
        category_vfgs = (
            set(group["VFG_ID"])
            & background_vfgs
        )

        background_with_category = len(
            category_vfgs
        )

        study_matching_vfgs = (
            category_vfgs
            & study_vfgs
        )

        study_with_category = len(
            study_matching_vfgs
        )

        if study_with_category < min_study_count:
            continue

        study_without_category = (
            study_size
            - study_with_category
        )

        background_nonstudy_with_category = (
            background_with_category
            - study_with_category
        )

        background_nonstudy_without_category = (
            background_size
            - study_size
            - background_nonstudy_with_category
        )

        contingency = [
            [
                study_with_category,
                study_without_category,
            ],
            [
                background_nonstudy_with_category,
                background_nonstudy_without_category,
            ],
        ]

        if min(
            value
            for row in contingency
            for value in row
        ) < 0:
            raise RuntimeError(
                f"Invalid contingency table for category '{category}': "
                f"{contingency}"
            )

        odds_ratio, p_value = fisher_exact(
            contingency,
            alternative="greater",
        )

        study_fraction = (
            study_with_category
            / study_size
        )

        background_fraction = (
            background_with_category
            / background_size
        )

        enrichment_ratio = (
            study_fraction
            / background_fraction
            if background_fraction > 0
            else np.nan
        )

        category_rows = group[
            group["VFG_ID"].isin(
                study_matching_vfgs
            )
        ]

        vfc_ids = sorted(
            category_rows["VFC_ID"]
            .dropna()
            .astype(str)
            .unique()
        )

        vf_names = sorted(
            category_rows["VF_name"]
            .dropna()
            .astype(str)
            .unique()
        )

        vf_ids = sorted(
            category_rows["VF_ID"]
            .dropna()
            .astype(str)
            .unique()
        )

        results.append({
            "VF_category": category,
            "VFC_ID": ";".join(vfc_ids),
            "study_count": study_with_category,
            "study_total_annotated": study_size,
            "background_count": background_with_category,
            "background_total_annotated": background_size,
            "study_fraction": study_fraction,
            "background_fraction": background_fraction,
            "enrichment_ratio": enrichment_ratio,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "study_VFG_IDs": ";".join(
                sorted(study_matching_vfgs)
            ),
            "study_VF_IDs": ";".join(vf_ids),
            "study_VF_names": ";".join(vf_names),
        })

    results = pd.DataFrame(results)

    if results.empty:
        return results

    _, adjusted, _, _ = multipletests(
        results["p_value"],
        method="fdr_bh",
    )

    results["fdr_bh"] = adjusted

    results["significant_fdr"] = (
        results["fdr_bh"]
        <= fdr_threshold
    )

    results["minus_log10_fdr"] = (
        -np.log10(
            results["fdr_bh"].clip(
                lower=np.nextafter(0, 1)
            )
        )
    )

    return results.sort_values(
        [
            "fdr_bh",
            "p_value",
            "enrichment_ratio",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


# ============================================================
# PLOT
# ============================================================

def wrap_label(text, width=50):
    words = str(text).split()
    lines = []
    current = []

    for word in words:
        candidate = " ".join(
            current + [word]
        )

        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(
                    " ".join(current)
                )
            current = [word]

    if current:
        lines.append(
            " ".join(current)
        )

    return "\n".join(lines)


def plot_enrichment(
    results,
    output_dir,
    top_n,
):
    if results.empty:
        return

    significant = results[
        results["significant_fdr"]
    ].copy()

    if not significant.empty:
        plot_df = significant.head(
            top_n
        ).copy()

        subtitle = (
            "FDR-significant VFDB categories"
        )
    else:
        plot_df = results.head(
            top_n
        ).copy()

        subtitle = (
            "Top VFDB categories; none passed "
            "the FDR threshold"
        )

    plot_df = plot_df.sort_values(
        [
            "minus_log10_fdr",
            "enrichment_ratio",
        ],
        ascending=True,
    )

    labels = []

    for _, row in plot_df.iterrows():
        label = wrap_label(
            row["VF_category"]
        )

        if row["VFC_ID"]:
            label += f" ({row['VFC_ID']})"

        labels.append(label)

    figure_height = max(
        5,
        0.58 * len(plot_df) + 1.8,
    )

    fig, ax = plt.subplots(
        figsize=(11, figure_height)
    )

    positions = np.arange(
        len(plot_df)
    )

    ax.barh(
        positions,
        plot_df["minus_log10_fdr"],
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "−log10(FDR-adjusted p-value)"
    )

    ax.set_ylabel(
        "VFDB category"
    )

    ax.set_title(
        f"VFDB category enrichment\n{subtitle}"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "VF_category_enrichment.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(
            output_dir,
            "VF_category_enrichment.pdf",
        ),
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read annotated nonredundant VF background
    # --------------------------------------------------------

    background = prepare_background(
        args.background
    )

    print(
        f"Background nonredundant VFs: "
        f"{background['VFG_ID'].nunique()}"
    )

    # --------------------------------------------------------
    # Read and match foreground/study list
    # --------------------------------------------------------

    study_values = read_study_list(
        args.study
    )

    study, unmatched = match_study_to_background(
        background=background,
        study_values=study_values,
    )

    pd.DataFrame({
        "unmatched_input_ID": unmatched
    }).to_csv(
        os.path.join(
            args.output,
            "unmatched_study_IDs.csv",
        ),
        index=False,
    )

    print(
        f"Foreground VFs matched to background: "
        f"{study['VFG_ID'].nunique()}"
    )

    if study.empty:
        raise RuntimeError(
            "No foreground IDs matched the annotated background table."
        )

    # --------------------------------------------------------
    # Build category mapping directly from background columns
    # --------------------------------------------------------

    category_mapping = build_category_mapping(
        background=background,
        include_unmatched_status=args.include_unmatched_status,
    )

    category_mapping.to_csv(
        os.path.join(
            args.output,
            "nonredundant_VF_category_annotations_long.csv",
        ),
        index=False,
    )

    annotated_background_ids = set(
        category_mapping["VFG_ID"]
    )

    annotated_study_ids = (
        set(study["VFG_ID"])
        & annotated_background_ids
    )

    annotated_background = background[
        background["VFG_ID"].isin(
            annotated_background_ids
        )
    ].copy()

    annotated_study = study[
        study["VFG_ID"].isin(
            annotated_study_ids
        )
    ].copy()

    background_without_category = background[
        ~background["VFG_ID"].isin(
            annotated_background_ids
        )
    ].copy()

    study_without_category = study[
        ~study["VFG_ID"].isin(
            annotated_study_ids
        )
    ].copy()

    background_without_category.to_csv(
        os.path.join(
            args.output,
            "nonredundant_VFs_without_category.csv",
        ),
        index=False,
    )

    study_without_category.to_csv(
        os.path.join(
            args.output,
            "study_VFs_without_category.csv",
        ),
        index=False,
    )

    print(
        f"Distinct VFDB categories: "
        f"{category_mapping['VF_category'].nunique()}"
    )

    print(
        f"Background VFs with usable category: "
        f"{len(annotated_background_ids)}/"
        f"{background['VFG_ID'].nunique()}"
    )

    print(
        f"Foreground VFs with usable category: "
        f"{len(annotated_study_ids)}/"
        f"{study['VFG_ID'].nunique()}"
    )

    # --------------------------------------------------------
    # Enrichment
    # --------------------------------------------------------

    results = perform_enrichment(
        annotated_background=annotated_background,
        annotated_study=annotated_study,
        category_mapping=category_mapping,
        fdr_threshold=args.fdr,
        min_study_count=args.min_category_count,
    )

    if results.empty:
        print(
            "No VFDB categories met the minimum annotated "
            "foreground count."
        )
        return

    results.to_csv(
        os.path.join(
            args.output,
            "VF_category_enrichment_all.csv",
        ),
        index=False,
    )

    significant = results[
        results["significant_fdr"]
    ].copy()

    significant.to_csv(
        os.path.join(
            args.output,
            "VF_category_enrichment_significant.csv",
        ),
        index=False,
    )

    plot_enrichment(
        results=results,
        output_dir=args.output,
        top_n=args.top_categories,
    )

    print()
    print("=" * 68)
    print("VFDB CATEGORY ENRICHMENT COMPLETE")
    print("=" * 68)

    print(
        f"Foreground VFs originally matched: "
        f"{study['VFG_ID'].nunique()}"
    )

    print(
        f"Foreground VFs used in Fisher tests: "
        f"{annotated_study['VFG_ID'].nunique()}"
    )

    print(
        f"Background VFs originally loaded: "
        f"{background['VFG_ID'].nunique()}"
    )

    print(
        f"Background VFs used in Fisher tests: "
        f"{annotated_background['VFG_ID'].nunique()}"
    )

    print(
        f"VFDB categories tested: "
        f"{len(results)}"
    )

    print(
        f"Significant VFDB categories: "
        f"{len(significant)}"
    )

    print()
    print(
        f"Outputs written to:\n"
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
