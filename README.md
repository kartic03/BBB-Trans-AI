# BBB-TransAI

A hybrid sequence-structure machine-learning framework for predicting blood-brain barrier (BBB)-penetrant peptides.

BBB-TransAI integrates sequence-derived descriptors (computed via iFeature) with AlphaFold3-predicted structural features to train a Random Forest classifier that distinguishes BBB-permeant (BBB+) from BBB-non-permeant (BBB-) peptides.

## Repository Structure

```
BBB-Trans-AI/
├── data/
│   ├── bbb_dataset.csv                  # Curated dataset of 538 peptides with BBB labels
│   └── bbb_struct_subset_100.csv        # 100-peptide mechanistic subset for structural analysis
├── features/
│   ├── bbb_all_features_11desc.csv      # Merged 1133 sequence descriptors per peptide
│   └── bbb_struct_features_100.csv      # Structural descriptors from the AlphaFold3 models
├── models/
│   ├── bbb_rf_top30.pkl                 # Hybrid model served by the application (30 features)
│   ├── bbb_rf_top30_features.json       # Its feature list
│   ├── bbb_rf_top200.pkl                # Sequence-only Random Forest (top-200 features)
│   └── bbb_rf_top200_features.json      # List of top-200 selected feature names
├── notebooks/
│   ├── 01_train_BBB_classifier.ipynb    # Model training, feature selection, cross-validation
│   ├── 03_hybrid_model.ipynb            # Hybrid model with structural features
│   └── structural_features.ipynb        # AlphaFold3 structural feature extraction
├── web/                                 # Interface: index.html, styles.css, app.js, icon.svg
├── bbb_core.py                          # Descriptors, model, prediction, contributions
├── app.py                               # Web application (Python standard library only)
├── requirements.txt                     # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kartic03/BBB-Trans-AI.git
   cd BBB-Trans-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install iFeature, which computes the sequence descriptors. Download it from
   [Superzchen/iFeature](https://github.com/Superzchen/iFeature) and place it so that
   `iFeature-master/iFeature-master/iFeature.py` sits next to `app.py`. The application
   starts without it and says so, but predictions need it.

4. Start the application:
   ```bash
   python app.py
   ```
   It listens on <http://localhost:8501>. Set `BBB_PORT` or `BBB_HOST` to change that.

The interface is served from `web/` and talks to a small JSON API in the same process.
Nothing beyond the packages above is required: no web framework, no build step, no
network access at runtime.

## Usage

### Web application

Enter one peptide or paste a batch. Batch input accepts `ID SEQUENCE` per line, FASTA, or
bare sequences, which are numbered automatically.

Each result gives the permeability probability, the call at the 0.5 cut-off, the peptide's
composition with percentiles against the 538 training peptides, and the descriptors that
produced the number. Those contributions are exact rather than approximate: each one is the
mean change in predicted probability across the 500 trees that split on that descriptor, and
they sum from the forest's base rate to the reported probability. Batch runs are ranked and
export to CSV.

### Structural descriptors

Three of the thirty model inputs come from the three-dimensional structure: `radius_gyration`,
`compactness` and `frac_positive`. Together they carry 10.2% of the forest's importance.

- `frac_positive` is computed exactly from the sequence.
- `radius_gyration` is measured from an attached PDB or mmCIF model when one is supplied.
  Otherwise it is estimated as `Rg = 2.2339 · L^0.5029`, a power law fitted on the 99
  reference structures in `features/bbb_struct_features_100.csv` (mean absolute error 1.16 Å,
  about 10% of the typical value).
- `compactness` follows as chain length divided by Rg.

Measured against those 99 peptides, estimating the two shape descriptors this way moves the
predicted probability by 0.018 on average and changes no call at the 0.5 cut-off. Attaching a
structure removes the estimate. Each result is labelled with which of the two paths produced it.

### Notebooks

Run the Jupyter notebooks in order to reproduce the full analysis pipeline:
1. `01_train_BBB_classifier.ipynb` - Train the sequence-only Random Forest classifier
2. `structural_features.ipynb` - Extract structural descriptors from AlphaFold3 models
3. `03_hybrid_model.ipynb` - Train the hybrid sequence-structure model

## Scope

Sequences use the twenty standard residues, 2 to 200 of them, up to 200 peptides per run.
Non-standard residues, modifications, D-amino acids, cyclisation and stapling fall outside the
training set, and a prediction for such a peptide reflects only its linear sequence. Research
use only.

## Citation

If you use BBB-TransAI in your research, please cite:

Kartic, Sharma A, Yi S, Park TS. Blood–brain barrier–transport artificial intelligence: A hybrid sequence–structure machine-learning framework for predicting blood–brain barrier-penetrant peptides. *Brain Network Disorders* (2026).
https://doi.org/10.1016/j.bnd.2026.04.001

```bibtex
@article{kartic2026bbbtransai,
  title   = {{Blood–brain barrier–transport artificial intelligence: A hybrid sequence–structure machine-learning framework for predicting blood–brain barrier-penetrant peptides}},
  author  = {Kartic and Sharma, Amitesh and Yi, Sanggyun and Park, Tae-Sik},
  journal = {Brain Network Disorders},
  year    = {2026},
  issn    = {3050-6239},
  doi     = {10.1016/j.bnd.2026.04.001},
  url     = {https://doi.org/10.1016/j.bnd.2026.04.001}
}
```

GitHub reads `CITATION.cff` in this repository, so the "Cite this repository"
button in the sidebar returns the same entry in APA or BibTeX.

## License

MIT. See [LICENSE](LICENSE).
