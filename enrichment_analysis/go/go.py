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
from tqdm import tqdm


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
    "locus_check/pick_VFs/significant_VFs.txt"
)

DEFAULT_OUTPUT = (
    "/data2/B_Li/vfdb/workflow_clade_translatorX_solved_dup_solved/"
    "locus_check/pick_VFs/GO"
)


# ============================================================
# UNIPROT SETTINGS
# ============================================================

# Pseudomonas aeruginosa PAO1 reference proteome
PAO1_PROTEOME = "UP000002438"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

UNIPROT_FIELDS = [
    "accession",
    "id",
    "gene_names",
    "gene_oln",
    "protein_name",
    "go_id",
    "go_p",
    "go_f",
    "go_c",
]

GO_NAMESPACES = {
    "BP": "Biological Process",
    "MF": "Molecular Function",
    "CC": "Cellular Component",
}


# ============================================================
# HELPERS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Perform GO enrichment using a nonredundant VF set "
            "as background and a selected VF list as foreground."
        )
    )

    parser.add_argument(
        "--background",
        default=DEFAULT_BACKGROUND,
        help="Nonredundant VF CSV.",
    )

    parser.add_argument(
        "--study",
        default=DEFAULT_STUDY,
        help=(
            "Foreground list with one VFG ID or locus tag per line. "
            "A CSV/TSV with VFG_ID, Accession, or locus_tag also works."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory.",
    )

    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="FDR significance threshold. Default: 0.05.",
    )

    parser.add_argument(
        "--min-go-count",
        type=int,
        default=2,
        help=(
            "Minimum number of foreground genes annotated to a GO term. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--top-terms",
        type=int,
        default=20,
        help="Number of GO terms shown in each plot.",
    )

    parser.add_argument(
        "--refresh-uniprot",
        action="store_true",
        help="Redownload PAO1 GO annotations instead of using cache.",
    )

    return parser.parse_args()


def extract_vfg(value):

    match = re.search(
        r"(VFG\d+)",
        str(value),
        re.IGNORECASE,
    )

    return match.group(1).upper() if match else None


def normalize_locus_tag(value):

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    match = re.search(r"\bPA\d{4}\b", value)

    return match.group(0) if match else None


def split_go_field(value):
    """
    Extract GO IDs and names from UniProt fields such as:

    pathogenesis [GO:0009405]
    protein binding [GO:0005515]
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    annotations = []

    # UniProt separates multiple entries with semicolons.
    for item in value.split(";"):

        item = item.strip()

        match = re.search(
            r"(.+?)\s*\[(GO:\d{7})\]",
            item,
        )

        if match:

            go_name = match.group(1).strip()
            go_id = match.group(2)

            annotations.append(
                (go_id, go_name)
            )

        else:

            # Sometimes only GO IDs are returned.
            go_match = re.search(
                r"(GO:\d{7})",
                item,
            )

            if go_match:

                annotations.append(
                    (
                        go_match.group(1),
                        "",
                    )
                )

    return annotations


def read_study_list(path):
    """
    Read either:
      - plain text with one ID per line;
      - CSV/TSV containing VFG_ID, Accession, or locus_tag.
    """

    extension = os.path.splitext(path)[1].lower()

    if extension in {".csv", ".tsv"}:

        separator = "\t" if extension == ".tsv" else ","

        table = pd.read_csv(
            path,
            sep=separator,
        )

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
                if line.strip()
                and not line.startswith("#")
            ]

    return values


def fetch_uniprot_pao1_annotations(
    output_path,
    refresh=False,
):
    """
    Download PAO1 proteins and GO annotations from UniProt.
    """

    if (
        os.path.isfile(output_path)
        and not refresh
    ):

        print(
            f"Using cached UniProt annotations:\n"
            f"{output_path}"
        )

        return pd.read_csv(
            output_path,
            sep="\t",
            dtype=str,
        )

    print(
        "Downloading PAO1 GO annotations from UniProt..."
    )

    params = {
        "query": f"proteome:{PAO1_PROTEOME}",
        "format": "tsv",
        "fields": ",".join(UNIPROT_FIELDS),
        "size": 500,
    }

    all_pages = []
    url = UNIPROT_URL

    progress = tqdm(
        desc="Downloading UniProt",
        unit="protein",
    )

    while url:

        response = requests.get(
            url,
            params=params if url == UNIPROT_URL else None,
            timeout=120,
        )

        response.raise_for_status()

        lines = response.text.strip().splitlines()

        if lines:

            if not all_pages:

                all_pages.extend(lines)

            else:

                # Skip header for subsequent pages.
                all_pages.extend(lines[1:])

            progress.update(
                max(0, len(lines) - 1)
            )

        next_url = None

        link_header = response.headers.get(
            "Link",
            "",
        )

        for part in link_header.split(","):

            if 'rel="next"' in part:

                match = re.search(
                    r"<([^>]+)>",
                    part,
                )

                if match:
                    next_url = match.group(1)

        url = next_url

        # Parameters are already encoded in next-page links.
        params = None

        if url:
            time.sleep(0.1)

    progress.close()

    if not all_pages:

        raise RuntimeError(
            "UniProt returned no PAO1 annotation records."
        )

    with open(output_path, "w") as handle:

        handle.write(
            "\n".join(all_pages) + "\n"
        )

    return pd.read_csv(
        output_path,
        sep="\t",
        dtype=str,
    )


def detect_uniprot_columns(uniprot):
    """
    UniProt column labels can change slightly, so identify them
    by their names rather than fixed positions.
    """

    columns = {
        str(column).lower(): column
        for column in uniprot.columns
    }

    def find_column(required_words):

        for lowercase, original in columns.items():

            if all(
                word in lowercase
                for word in required_words
            ):
                return original

        return None

    detected = {
        "accession": find_column(["entry"]),
        "gene_names": find_column(["gene names"]),
        "ordered_locus": (
            find_column(["ordered locus"])
            or find_column(["gene names", "ordered"])
        ),
        "protein_name": find_column(["protein names"]),
        "go_bp": (
            find_column(["gene ontology", "biological process"])
            or find_column(["go", "biological process"])
        ),
        "go_mf": (
            find_column(["gene ontology", "molecular function"])
            or find_column(["go", "molecular function"])
        ),
        "go_cc": (
            find_column(["gene ontology", "cellular component"])
            or find_column(["go", "cellular component"])
        ),
        "go_ids": (
            find_column(["gene ontology ids"])
            or find_column(["go ids"])
        ),
    }

    return detected


def build_uniprot_locus_mapping(uniprot):
    """
    Convert downloaded UniProt annotations into one row per
    PAO1 locus-tag/GO-term combination.
    """

    detected = detect_uniprot_columns(
        uniprot
    )

    if detected["ordered_locus"] is None:

        raise ValueError(
            "Could not identify the ordered-locus-name column "
            f"in UniProt output. Columns were:\n"
            f"{list(uniprot.columns)}"
        )

    mapping_rows = []

    namespace_columns = {
        "BP": detected["go_bp"],
        "MF": detected["go_mf"],
        "CC": detected["go_cc"],
    }

    for _, row in tqdm(
        uniprot.iterrows(),
        total=len(uniprot),
        desc="Parsing GO annotations",
        unit="protein",
    ):

        ordered_locus_value = row[
            detected["ordered_locus"]
        ]

        locus_tags = set(
            re.findall(
                r"\bPA\d{4}\b",
                str(ordered_locus_value).upper(),
            )
        )

        if not locus_tags:
            continue

        for namespace, column in namespace_columns.items():

            if column is None:
                continue

            go_annotations = split_go_field(
                row[column]
            )

            for locus_tag in locus_tags:

                for go_id, go_name in go_annotations:

                    mapping_rows.append({
                        "locus_tag": locus_tag,
                        "go_id": go_id,
                        "go_name": go_name,
                        "namespace": namespace,
                        "namespace_name": (
                            GO_NAMESPACES[namespace]
                        ),
                        "uniprot_accession": (
                            row.get(
                                detected["accession"],
                                "",
                            )
                            if detected["accession"]
                            else ""
                        ),
                        "uniprot_gene_names": (
                            row.get(
                                detected["gene_names"],
                                "",
                            )
                            if detected["gene_names"]
                            else ""
                        ),
                        "uniprot_protein_name": (
                            row.get(
                                detected["protein_name"],
                                "",
                            )
                            if detected["protein_name"]
                            else ""
                        ),
                    })

    mapping = pd.DataFrame(
        mapping_rows
    )

    if mapping.empty:

        raise RuntimeError(
            "No PAO1 locus-tag to GO mappings were parsed."
        )

    return mapping.drop_duplicates(
        subset=[
            "locus_tag",
            "go_id",
            "namespace",
        ]
    )


def perform_enrichment(
    background_df,
    study_df,
    go_mapping,
    fdr_threshold,
    min_study_count,
):
    """
    Fisher exact over-representation analysis.

    Unit of analysis: one retained VF/locus tag.
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

    background_size = len(
        background_loci
    )

    study_size = len(
        study_loci
    )

    results = []

    for (
        namespace,
        go_id,
        go_name,
    ), group in go_mapping.groupby(
        [
            "namespace",
            "go_id",
            "go_name",
        ],
        dropna=False,
    ):

        annotated_loci = set(
            group["locus_tag"]
        ) & background_loci

        background_with_term = len(
            annotated_loci
        )

        study_with_term = len(
            annotated_loci & study_loci
        )

        if study_with_term < min_study_count:
            continue

        study_without_term = (
            study_size - study_with_term
        )

        background_nonstudy_with_term = (
            background_with_term
            - study_with_term
        )

        background_nonstudy_without_term = (
            background_size
            - study_size
            - background_nonstudy_with_term
        )

        contingency = [
            [
                study_with_term,
                study_without_term,
            ],
            [
                background_nonstudy_with_term,
                background_nonstudy_without_term,
            ],
        ]

        odds_ratio, p_value = fisher_exact(
            contingency,
            alternative="greater",
        )

        study_fraction = (
            study_with_term / study_size
            if study_size
            else np.nan
        )

        background_fraction = (
            background_with_term / background_size
            if background_size
            else np.nan
        )

        enrichment_ratio = (
            study_fraction / background_fraction
            if background_fraction > 0
            else np.nan
        )

        matching_study_loci = sorted(
            annotated_loci & study_loci
        )

        matching_vfgs = sorted(
            study_df.loc[
                study_df["locus_tag"].isin(
                    matching_study_loci
                ),
                "VFG_ID",
            ].dropna().unique()
        )

        results.append({
            "namespace": namespace,
            "namespace_name": (
                GO_NAMESPACES[namespace]
            ),
            "go_id": go_id,
            "go_name": go_name,
            "study_count": study_with_term,
            "study_total": study_size,
            "background_count": (
                background_with_term
            ),
            "background_total": (
                background_size
            ),
            "study_fraction": study_fraction,
            "background_fraction": (
                background_fraction
            ),
            "enrichment_ratio": (
                enrichment_ratio
            ),
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "study_locus_tags": ";".join(
                matching_study_loci
            ),
            "study_VFG_IDs": ";".join(
                matching_vfgs
            ),
        })

    results = pd.DataFrame(
        results
    )

    if results.empty:
        return results

    # Correct separately within BP, MF, and CC.
    corrected_frames = []

    for namespace, group in results.groupby(
        "namespace"
    ):

        group = group.copy()

        _, adjusted, _, _ = multipletests(
            group["p_value"],
            method="fdr_bh",
        )

        group["fdr_bh"] = adjusted

        group["significant_fdr"] = (
            group["fdr_bh"]
            <= fdr_threshold
        )

        corrected_frames.append(
            group
        )

    results = pd.concat(
        corrected_frames,
        ignore_index=True,
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
            "namespace",
            "fdr_bh",
            "p_value",
        ]
    )


def plot_namespace(
    results,
    namespace,
    output_dir,
    top_n,
):
    """
    Plot top enriched terms for one GO namespace.
    """

    subset = results[
        results["namespace"] == namespace
    ].copy()

    if subset.empty:
        return

    significant = subset[
        subset["significant_fdr"]
    ].copy()

    if not significant.empty:
        plot_df = significant.head(
            top_n
        ).copy()
    else:
        plot_df = subset.head(
            top_n
        ).copy()

    plot_df = plot_df.sort_values(
        [
            "minus_log10_fdr",
            "enrichment_ratio",
        ],
        ascending=True,
    )

    labels = (
        plot_df["go_name"]
        .fillna("")
        .astype(str)
        + " ("
        + plot_df["go_id"]
        + ")"
    )

    figure_height = max(
        5,
        0.42 * len(plot_df) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(10, figure_height)
    )

    ax.barh(
        np.arange(len(plot_df)),
        plot_df["minus_log10_fdr"],
    )

    ax.set_yticks(
        np.arange(len(plot_df))
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "−log10(FDR-adjusted p-value)"
    )

    ax.set_ylabel(
        GO_NAMESPACES[namespace]
    )

    significance_label = (
        "FDR-significant terms"
        if not significant.empty
        else "Top terms; none passed FDR threshold"
    )

    ax.set_title(
        f"GO {GO_NAMESPACES[namespace]} enrichment\n"
        f"{significance_label}"
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
            f"GO_enrichment_{namespace}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(
            output_dir,
            f"GO_enrichment_{namespace}.pdf",
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

    unmatched = []

    matched_original = set(
        study["VFG_ID"]
    ) | set(
        study["locus_tag"]
    )

    for value in study_values:

        vfg = extract_vfg(value)
        locus = normalize_locus_tag(value)

        if (
            vfg not in matched_original
            and locus not in matched_original
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

    if study.empty:

        raise RuntimeError(
            "No foreground IDs matched the "
            "nonredundant background table."
        )

    # --------------------------------------------------------
    # Download and parse PAO1 GO annotations
    # --------------------------------------------------------

    uniprot_cache = os.path.join(
        args.output,
        "PAO1_UniProt_GO_annotations.tsv",
    )

    uniprot = fetch_uniprot_pao1_annotations(
        output_path=uniprot_cache,
        refresh=args.refresh_uniprot,
    )

    go_mapping = build_uniprot_locus_mapping(
        uniprot
    )

    go_mapping.to_csv(
        os.path.join(
            args.output,
            "PAO1_locus_tag_to_GO.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Annotate background VF table
    # --------------------------------------------------------

    annotated_background = background.merge(
        go_mapping,
        on="locus_tag",
        how="left",
    )

    annotated_background.to_csv(
        os.path.join(
            args.output,
            "nonredundant_VF_GO_annotations_long.csv",
        ),
        index=False,
    )

    mapped_background = set(
        annotated_background.loc[
            annotated_background["go_id"].notna(),
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
            "nonredundant_VFs_without_GO.csv",
        ),
        index=False,
    )

    print(
        f"Background loci with GO annotation: "
        f"{len(mapped_background)}/"
        f"{background['locus_tag'].nunique()}"
    )

    # --------------------------------------------------------
    # Enrichment
    # --------------------------------------------------------

    results = perform_enrichment(
        background_df=background,
        study_df=study,
        go_mapping=go_mapping,
        fdr_threshold=args.fdr,
        min_study_count=args.min_go_count,
    )

    if results.empty:

        print(
            "No GO terms met the minimum foreground count."
        )

        return

    results.to_csv(
        os.path.join(
            args.output,
            "GO_enrichment_all.csv",
        ),
        index=False,
    )

    for namespace in [
        "BP",
        "MF",
        "CC",
    ]:

        namespace_results = results[
            results["namespace"] == namespace
        ].copy()

        namespace_results.to_csv(
            os.path.join(
                args.output,
                f"GO_enrichment_{namespace}.csv",
            ),
            index=False,
        )

        significant = namespace_results[
            namespace_results["significant_fdr"]
        ]

        significant.to_csv(
            os.path.join(
                args.output,
                f"GO_enrichment_{namespace}_significant.csv",
            ),
            index=False,
        )

        plot_namespace(
            results=results,
            namespace=namespace,
            output_dir=args.output,
            top_n=args.top_terms,
        )

    print()
    print("=" * 64)
    print("GO ENRICHMENT COMPLETE")
    print("=" * 64)

    print(
        f"Foreground loci tested: "
        f"{study['locus_tag'].nunique()}"
    )

    print(
        f"Background loci: "
        f"{background['locus_tag'].nunique()}"
    )

    print(
        f"Significant BP terms: "
        f"{len(results[(results['namespace'] == 'BP') & results['significant_fdr']])}"
    )

    print(
        f"Significant MF terms: "
        f"{len(results[(results['namespace'] == 'MF') & results['significant_fdr']])}"
    )

    print(
        f"Significant CC terms: "
        f"{len(results[(results['namespace'] == 'CC') & results['significant_fdr']])}"
    )

    print()
    print(
        f"Outputs written to:\n"
        f"{args.output}"
    )


if __name__ == "__main__":
    main()