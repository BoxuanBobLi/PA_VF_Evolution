#!/usr/bin/env python3

import os
import re
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_BACKGROUND = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/output/"
    "core_VF_nonredundant_representatives.csv"
)

DEFAULT_STUDY = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/GO/study/study_BB_vs_BA.txt"
)

DEFAULT_OUTPUT = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/KEGG/study_BB_vs_BA"
)


# ============================================================
# KEGG SETTINGS
# ============================================================

# KEGG organism code for Pseudomonas aeruginosa PAO1
KEGG_ORGANISM = "pae"
KEGG_REST = "https://rest.kegg.jp"

# KEGG recommends no more than 3 API requests per second.
REQUEST_DELAY = 0.4

# Broad overview maps can dominate enrichment results.
# They are retained by default, but can be removed with
# --exclude-overview-pathways.
OVERVIEW_PATHWAY_NUMBERS = {
    "01100",  # Metabolic pathways
    "01110",  # Biosynthesis of secondary metabolites
    "01120",  # Microbial metabolism in diverse environments
    "01200",  # Carbon metabolism
    "01210",  # 2-Oxocarboxylic acid metabolism
    "01212",  # Fatty acid metabolism
    "01230",  # Biosynthesis of amino acids
    "01232",  # Nucleotide metabolism
    "01240",  # Biosynthesis of cofactors
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Perform KEGG pathway over-representation analysis using "
            "a nonredundant VF set as background and a selected VF list "
            "as foreground."
        )
    )

    parser.add_argument(
        "--background",
        default=DEFAULT_BACKGROUND,
        help=(
            "Nonredundant VF CSV containing VFG_ID and locus_tag. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--study",
        default=DEFAULT_STUDY,
        help=(
            "Foreground list containing one VFG ID or PAO1 locus tag per "
            "line. CSV/TSV files containing VFG_ID, Accession, or locus_tag "
            "also work. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory. Default: %(default)s",
    )

    parser.add_argument(
        "--organism",
        default=KEGG_ORGANISM,
        help=(
            "KEGG organism code. PAO1 is pae. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR threshold. Default: %(default)s",
    )

    parser.add_argument(
        "--min-pathway-count",
        type=int,
        default=2,
        help=(
            "Minimum number of foreground loci assigned to a pathway "
            "for that pathway to be tested. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--top-pathways",
        type=int,
        default=20,
        help="Maximum number of pathways shown in the plot. Default: %(default)s",
    )

    parser.add_argument(
        "--refresh-kegg",
        action="store_true",
        help="Redownload KEGG mappings instead of using cached files.",
    )

    parser.add_argument(
        "--exclude-overview-pathways",
        action="store_true",
        help=(
            "Exclude broad KEGG overview pathways such as pae01100, "
            "pae01110, and pae01120."
        ),
    )

    return parser.parse_args()


# ============================================================
# ID HELPERS
# ============================================================

def extract_vfg(value):
    match = re.search(r"(VFG\d+)", str(value), re.IGNORECASE)
    return match.group(1).upper() if match else None


def normalize_locus_tag(value):
    if pd.isna(value):
        return None

    match = re.search(r"\bPA\d{4}\b", str(value).upper())
    return match.group(0) if match else None


def normalize_kegg_gene(value, organism):
    """
    Convert values such as:
        pae:PA0001
        PA0001
    into:
        pae:PA0001
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(
        rf"\b{re.escape(organism)}:(PA\d{{4}})\b",
        text,
        re.IGNORECASE,
    )

    if match:
        return f"{organism}:{match.group(1).upper()}"

    locus = normalize_locus_tag(text)

    if locus:
        return f"{organism}:{locus}"

    return None


def pathway_number(pathway_id):
    """
    Extract 5-digit pathway number from IDs such as pae02025.
    """
    match = re.search(r"(\d{5})$", str(pathway_id))
    return match.group(1) if match else None


# ============================================================
# INPUT
# ============================================================

def read_study_list(path):
    """
    Read either:
      - plain text with one ID per line;
      - CSV/TSV containing VFG_ID, Accession, or locus_tag.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension in {".csv", ".tsv"}:
        separator = "\t" if extension == ".tsv" else ","
        table = pd.read_csv(path, sep=separator)

        preferred_columns = [
            "VFG_ID",
            "Accession",
            "locus_tag",
        ]

        selected_column = None

        for column in preferred_columns:
            if column in table.columns:
                selected_column = column
                break

        if selected_column is None:
            selected_column = table.columns[0]

        values = (
            table[selected_column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    else:
        with open(path, "r") as handle:
            values = [
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            ]

    return values


# ============================================================
# KEGG DOWNLOAD
# ============================================================

def request_kegg_text(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.text


def fetch_kegg_pathway_names(
    organism,
    cache_path,
    refresh=False,
):
    """
    Download organism-specific KEGG pathway names.

    Endpoint:
        /list/pathway/<organism>
    """
    if os.path.isfile(cache_path) and not refresh:
        print(f"Using cached KEGG pathway names:\n{cache_path}")
        return pd.read_csv(cache_path, sep="\t", dtype=str)

    url = f"{KEGG_REST}/list/pathway/{organism}"

    print(f"Downloading KEGG pathway names:\n{url}")

    text = request_kegg_text(url)
    rows = []

    for line in text.splitlines():
        if not line.strip():
            continue

        fields = line.split("\t", 1)

        if len(fields) != 2:
            continue

        pathway_id = fields[0].replace("path:", "").strip()
        pathway_name = fields[1].strip()

        rows.append({
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
        })

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError(
            f"No pathway names were returned for KEGG organism {organism}."
        )

    table.to_csv(cache_path, sep="\t", index=False)
    return table


def fetch_kegg_gene_pathway_links(
    organism,
    cache_path,
    refresh=False,
):
    """
    Download all organism-gene to organism-pathway links.

    Endpoint:
        /link/pathway/<organism>

    Output columns:
        kegg_gene_id, locus_tag, pathway_id
    """
    if os.path.isfile(cache_path) and not refresh:
        print(f"Using cached KEGG gene-pathway links:\n{cache_path}")
        return pd.read_csv(cache_path, sep="\t", dtype=str)

    url = f"{KEGG_REST}/link/pathway/{organism}"

    print(f"Downloading KEGG gene-pathway links:\n{url}")

    text = request_kegg_text(url)
    rows = []

    for line in text.splitlines():
        if not line.strip():
            continue

        fields = line.split("\t")

        if len(fields) < 2:
            continue

        left = fields[0].strip()
        right = fields[1].strip()

        # KEGG currently returns gene first and pathway second for this call,
        # but detect the fields rather than relying solely on their positions.
        values = [left, right]

        gene_id = next(
            (
                value
                for value in values
                if value.lower().startswith(f"{organism.lower()}:")
            ),
            None,
        )

        pathway_id = next(
            (
                value.replace("path:", "")
                for value in values
                if value.lower().startswith("path:")
            ),
            None,
        )

        if gene_id is None or pathway_id is None:
            continue

        locus_tag = normalize_locus_tag(gene_id)

        if locus_tag is None:
            continue

        rows.append({
            "kegg_gene_id": normalize_kegg_gene(gene_id, organism),
            "locus_tag": locus_tag,
            "pathway_id": pathway_id,
        })

    table = pd.DataFrame(rows).drop_duplicates()

    if table.empty:
        raise RuntimeError(
            f"No gene-pathway links were returned for KEGG organism {organism}."
        )

    table.to_csv(cache_path, sep="\t", index=False)
    return table


def build_kegg_mapping(
    organism,
    output_dir,
    refresh=False,
    exclude_overview=False,
):
    pathway_names_path = os.path.join(
        output_dir,
        f"{organism}_KEGG_pathway_names.tsv",
    )

    gene_links_path = os.path.join(
        output_dir,
        f"{organism}_KEGG_gene_pathway_links.tsv",
    )

    names = fetch_kegg_pathway_names(
        organism=organism,
        cache_path=pathway_names_path,
        refresh=refresh,
    )

    links = fetch_kegg_gene_pathway_links(
        organism=organism,
        cache_path=gene_links_path,
        refresh=refresh,
    )

    mapping = links.merge(
        names,
        on="pathway_id",
        how="left",
    )

    missing_names = mapping["pathway_name"].isna().sum()

    if missing_names:
        print(
            f"Warning: {missing_names} gene-pathway rows lacked a "
            "matching pathway name."
        )

    mapping["pathway_number"] = (
        mapping["pathway_id"]
        .apply(pathway_number)
    )

    if exclude_overview:
        before = mapping["pathway_id"].nunique()

        mapping = mapping[
            ~mapping["pathway_number"].isin(
                OVERVIEW_PATHWAY_NUMBERS
            )
        ].copy()

        after = mapping["pathway_id"].nunique()

        print(
            f"Excluded {before - after} broad overview pathways."
        )

    mapping = mapping.drop_duplicates(
        subset=[
            "locus_tag",
            "pathway_id",
        ]
    )

    mapping.to_csv(
        os.path.join(
            output_dir,
            f"{organism}_locus_tag_to_KEGG_pathway.csv",
        ),
        index=False,
    )

    return mapping


# ============================================================
# ENRICHMENT
# ============================================================

def perform_enrichment(
    background_df,
    study_df,
    pathway_mapping,
    fdr_threshold,
    min_study_count,
):
    """
    One-sided Fisher exact over-representation analysis.

    Unit of analysis:
        one retained nonredundant VF/locus tag

    Background:
        all retained VF loci that were eligible for study selection

    Foreground:
        selected study loci

    For each pathway, the 2x2 table is:

                          In pathway    Not in pathway
        Study                 a               b
        Background-study      c               d
    """
    background_loci = set(
        background_df["locus_tag"]
        .dropna()
        .unique()
    )

    study_loci = set(
        study_df["locus_tag"]
        .dropna()
        .unique()
    )

    study_loci &= background_loci

    background_size = len(background_loci)
    study_size = len(study_loci)

    results = []

    for (
        pathway_id,
        pathway_name,
    ), group in pathway_mapping.groupby(
        [
            "pathway_id",
            "pathway_name",
        ],
        dropna=False,
    ):
        annotated_loci = (
            set(group["locus_tag"])
            & background_loci
        )

        background_with_pathway = len(
            annotated_loci
        )

        study_matching_loci = (
            annotated_loci
            & study_loci
        )

        study_with_pathway = len(
            study_matching_loci
        )

        if study_with_pathway < min_study_count:
            continue

        study_without_pathway = (
            study_size
            - study_with_pathway
        )

        background_nonstudy_with_pathway = (
            background_with_pathway
            - study_with_pathway
        )

        background_nonstudy_without_pathway = (
            background_size
            - study_size
            - background_nonstudy_with_pathway
        )

        if min(
            study_with_pathway,
            study_without_pathway,
            background_nonstudy_with_pathway,
            background_nonstudy_without_pathway,
        ) < 0:
            raise RuntimeError(
                f"Invalid contingency table for {pathway_id}. "
                "Check foreground/background construction."
            )

        contingency = [
            [
                study_with_pathway,
                study_without_pathway,
            ],
            [
                background_nonstudy_with_pathway,
                background_nonstudy_without_pathway,
            ],
        ]

        odds_ratio, p_value = fisher_exact(
            contingency,
            alternative="greater",
        )

        study_fraction = (
            study_with_pathway / study_size
            if study_size
            else np.nan
        )

        background_fraction = (
            background_with_pathway / background_size
            if background_size
            else np.nan
        )

        enrichment_ratio = (
            study_fraction / background_fraction
            if background_fraction > 0
            else np.nan
        )

        matching_vfgs = sorted(
            study_df.loc[
                study_df["locus_tag"].isin(
                    study_matching_loci
                ),
                "VFG_ID",
            ]
            .dropna()
            .unique()
        )

        results.append({
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "study_count": study_with_pathway,
            "study_total": study_size,
            "background_count": background_with_pathway,
            "background_total": background_size,
            "study_fraction": study_fraction,
            "background_fraction": background_fraction,
            "enrichment_ratio": enrichment_ratio,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "study_locus_tags": ";".join(
                sorted(study_matching_loci)
            ),
            "study_VFG_IDs": ";".join(
                matching_vfgs
            ),
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

def wrap_label(text, width=55):
    words = str(text).split()
    lines = []
    current = []

    for word in words:
        candidate = " ".join(current + [word])

        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

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

        subtitle = "FDR-significant pathways"
    else:
        plot_df = results.head(
            top_n
        ).copy()

        subtitle = "Top pathways; none passed the FDR threshold"

    plot_df = plot_df.sort_values(
        [
            "minus_log10_fdr",
            "enrichment_ratio",
        ],
        ascending=True,
    )

    labels = [
        f"{wrap_label(name)} ({pathway_id})"
        for name, pathway_id in zip(
            plot_df["pathway_name"],
            plot_df["pathway_id"],
        )
    ]

    figure_height = max(
        5,
        0.55 * len(plot_df) + 1.8,
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
        "KEGG pathway"
    )

    ax.set_title(
        f"KEGG pathway enrichment\n{subtitle}"
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
            "KEGG_enrichment.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(
            output_dir,
            "KEGG_enrichment.pdf",
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
    # Read nonredundant VF background
    # --------------------------------------------------------

    background = pd.read_csv(
        args.background
    )

    required = {
        "VFG_ID",
        "locus_tag",
    }

    missing = required - set(
        background.columns
    )

    if missing:
        raise ValueError(
            "Background file is missing columns: "
            + ", ".join(sorted(missing))
        )

    background["VFG_ID"] = (
        background["VFG_ID"]
        .apply(extract_vfg)
    )

    background["locus_tag"] = (
        background["locus_tag"]
        .apply(normalize_locus_tag)
    )

    background = background[
        background["VFG_ID"].notna()
        & background["locus_tag"].notna()
    ].copy()

    background = background.drop_duplicates(
        subset=[
            "VFG_ID",
            "locus_tag",
        ]
    )

    print(
        f"Background VFs: "
        f"{background['VFG_ID'].nunique()}"
    )

    print(
        f"Background locus tags: "
        f"{background['locus_tag'].nunique()}"
    )

    # --------------------------------------------------------
    # Read foreground/study list
    # --------------------------------------------------------

    study_values = read_study_list(
        args.study
    )

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
        background["VFG_ID"].isin(
            study_vfgs
        )
        | background["locus_tag"].isin(
            study_loci
        )
    ].copy()

    matched_vfgs = set(
        study["VFG_ID"]
    )

    matched_loci = set(
        study["locus_tag"]
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
        f"Foreground VFs matched: "
        f"{study['VFG_ID'].nunique()}"
    )

    print(
        f"Foreground locus tags matched: "
        f"{study['locus_tag'].nunique()}"
    )

    if study.empty:
        raise RuntimeError(
            "No foreground IDs matched the "
            "nonredundant background table."
        )

    # --------------------------------------------------------
    # Download and parse KEGG pathway annotations
    # --------------------------------------------------------

    pathway_mapping = build_kegg_mapping(
        organism=args.organism,
        output_dir=args.output,
        refresh=args.refresh_kegg,
        exclude_overview=args.exclude_overview_pathways,
    )

    # --------------------------------------------------------
    # Annotate background VF table
    # --------------------------------------------------------

    annotated_background = background.merge(
        pathway_mapping,
        on="locus_tag",
        how="left",
    )

    annotated_background.to_csv(
        os.path.join(
            args.output,
            "nonredundant_VF_KEGG_annotations_long.csv",
        ),
        index=False,
    )

    mapped_background = set(
        annotated_background.loc[
            annotated_background["pathway_id"].notna(),
            "locus_tag",
        ]
    )

    unmapped_background = background[
        ~background["locus_tag"].isin(
            mapped_background
        )
    ].copy()

    unmapped_background.to_csv(
        os.path.join(
            args.output,
            "nonredundant_VFs_without_KEGG_pathway.csv",
        ),
        index=False,
    )

    mapped_study = set(
        study["locus_tag"]
    ) & mapped_background

    study_without_kegg = study[
        ~study["locus_tag"].isin(
            mapped_background
        )
    ].copy()

    study_without_kegg.to_csv(
        os.path.join(
            args.output,
            "study_VFs_without_KEGG_pathway.csv",
        ),
        index=False,
    )

    print(
        f"Background loci with KEGG pathway annotation: "
        f"{len(mapped_background)}/"
        f"{background['locus_tag'].nunique()}"
    )

    print(
        f"Foreground loci with KEGG pathway annotation: "
        f"{len(mapped_study)}/"
        f"{study['locus_tag'].nunique()}"
    )

    # --------------------------------------------------------
    # Enrichment
    # --------------------------------------------------------

    results = perform_enrichment(
        background_df=background,
        study_df=study,
        pathway_mapping=pathway_mapping,
        fdr_threshold=args.fdr,
        min_study_count=args.min_pathway_count,
    )

    if results.empty:
        print(
            "No KEGG pathways met the minimum foreground count."
        )
        return

    results.to_csv(
        os.path.join(
            args.output,
            "KEGG_enrichment_all.csv",
        ),
        index=False,
    )

    significant = results[
        results["significant_fdr"]
    ].copy()

    significant.to_csv(
        os.path.join(
            args.output,
            "KEGG_enrichment_significant.csv",
        ),
        index=False,
    )

    plot_enrichment(
        results=results,
        output_dir=args.output,
        top_n=args.top_pathways,
    )

    print()
    print("=" * 64)
    print("KEGG ENRICHMENT COMPLETE")
    print("=" * 64)

    print(
        f"Foreground loci tested: "
        f"{study['locus_tag'].nunique()}"
    )

    print(
        f"Background loci tested: "
        f"{background['locus_tag'].nunique()}"
    )

    print(
        f"KEGG pathways tested: "
        f"{len(results)}"
    )

    print(
        f"Significant KEGG pathways: "
        f"{len(significant)}"
    )

    print()
    print(
        f"Outputs written to:\n"
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
