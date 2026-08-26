# KEGG Enrichment Analysis

This analysis uses the **same study lists generated for the GO enrichment analysis**:

```text
study_AA_vs_AB.txt
study_BB_vs_BA.txt
```

These files contain the selected VFG IDs used as the foreground set. The KEGG script accepts the same foreground format as the GO workflow.

## Run KEGG enrichment

For Clade A:

```bash
python kegg.py \
  --study ../GO/study/study_AA_vs_AB.txt \
  --output study_AA_vs_AB
```

For Clade B:

```bash
python kegg.py \
  --study ../GO/study/study_BB_vs_BA.txt \
  --output study_BB_vs_BA
```

The background is the same nonredundant VF representative set used in the GO analysis.

## Main outputs

```text
KEGG_enrichment_all.csv
```

All tested KEGG pathway enrichment results.

```text
KEGG_enrichment_significant.csv
```

Pathways passing the FDR threshold.

```text
KEGG_enrichment.png
KEGG_enrichment.pdf
```

Plot of the top enriched KEGG pathways.

```text
pae_locus_tag_to_KEGG_pathway.csv
```

PAO1 locus-tag to KEGG pathway mapping.

```text
nonredundant_VF_KEGG_annotations_long.csv
```

KEGG pathway annotations for the nonredundant VF background.

```text
nonredundant_VFs_without_KEGG_pathway.csv
study_VFs_without_KEGG_pathway.csv
unmatched_study_IDs.csv
```

Files listing VFs or study IDs that could not be mapped to KEGG pathways or the background set.

KEGG enrichment is performed using a one-sided Fisher's exact test followed by Benjamini-Hochberg FDR correction.
