# GO Enrichment Analysis

## 1. Generate study lists

Run:

```bash
cd study
python study.py
```

This generates:

```text
study_AA_vs_AB.csv
study_AA_vs_AB.txt
study_BB_vs_BA.csv
study_BB_vs_BA.txt
```

The `.txt` files contain the selected VFG IDs used as foreground genes for GO enrichment.

The selected VFs include:

* top 15 genes with the largest dN/dS difference in each direction
* any genes with mean dN/dS > 1

## 2. Run GO enrichment

For Clade A:

```bash
python go.py \
  --study study/study_AA_vs_AB.txt \
  --output cladeA_output
```

For Clade B:

```bash
python go.py \
  --study study/study_BB_vs_BA.txt \
  --output cladeB_output
```

## Main outputs

```text
GO_enrichment_all.csv
```

All GO enrichment results.

```text
GO_enrichment_BP.csv
GO_enrichment_MF.csv
GO_enrichment_CC.csv
```

Results for Biological Process, Molecular Function, and Cellular Component.

```text
GO_enrichment_BP_significant.csv
GO_enrichment_MF_significant.csv
GO_enrichment_CC_significant.csv
```

Only GO terms passing the FDR threshold.

```text
GO_enrichment_BP.png / .pdf
GO_enrichment_MF.png / .pdf
GO_enrichment_CC.png / .pdf
```

Plots of the top enriched GO terms.

Other useful files:

```text
PAO1_locus_tag_to_GO.csv
```

PAO1 locus-tag to GO annotation mapping.

```text
nonredundant_VF_GO_annotations_long.csv
```

GO annotations for the nonredundant VF background.

```text
unmatched_study_IDs.csv
```

Study VFG IDs that could not be matched to the background set.
