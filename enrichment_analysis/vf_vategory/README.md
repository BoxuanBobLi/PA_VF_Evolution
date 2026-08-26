# VFDB Category Enrichment

This analysis uses the **same study lists generated for the GO enrichment analysis**:

```text
study_AA_vs_AB.txt
study_BB_vs_BA.txt
```

These files are used as the foreground VF sets.

## Run enrichment

For Clade A:

```bash
python vf_category.py \
  --study ../GO/study/study_AA_vs_AB.txt \
  --output study_AA_vs_AB
```

For Clade B:

```bash
python vf_category.py \
  --study ../GO/study/study_BB_vs_BA.txt \
  --output study_BB_vs_BA
```

The background is the nonredundant VF set annotated with VFDB categories.

## Main outputs

```text
VF_category_enrichment_all.csv
```

All tested VFDB category enrichment results.

```text
VF_category_enrichment_significant.csv
```

VFDB categories passing the FDR threshold.

```text
VF_category_enrichment.png
VF_category_enrichment.pdf
```

Plots of the top enriched VFDB categories.

```text
nonredundant_VF_category_annotations_long.csv
```

VFDB category annotations for the nonredundant background set.

```text
nonredundant_VFs_without_category.csv
study_VFs_without_category.csv
unmatched_study_IDs.csv
```

Lists of background or study VFs that could not be assigned to a VFDB category or matched to the background.

Enrichment is tested using a one-sided Fisher's exact test followed by Benjamini-Hochberg FDR correction.
