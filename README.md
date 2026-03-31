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
│   └── bbb_all_features_11desc.csv      # Merged 1133 sequence descriptors per peptide
├── models/
│   ├── bbb_rf_top200.pkl                # Trained Random Forest model (top-200 features)
│   └── bbb_rf_top200_features.json      # List of top-200 selected feature names
├── notebooks/
│   ├── 01_train_BBB_classifier.ipynb    # Model training, feature selection, cross-validation
│   ├── 03_hybrid_model.ipynb            # Hybrid model with structural features
│   └── structural_features.ipynb        # AlphaFold3 structural feature extraction
├── app.py                               # Streamlit web application
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

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Streamlit Application
The interactive web application allows users to input peptide sequences in "ID SEQUENCE" format. The model processes each entry and returns BBB permeability predictions (probability and binary label) in real time. Results are displayed as a ranked table with a color-scaled heatmap and bar plot. A CSV export function is included for downstream analysis.

### Notebooks
Run the Jupyter notebooks in order to reproduce the full analysis pipeline:
1. `01_train_BBB_classifier.ipynb` - Train the sequence-only Random Forest classifier
2. `structural_features.ipynb` - Extract structural descriptors from AlphaFold3 models
3. `03_hybrid_model.ipynb` - Train the hybrid sequence-structure model

## Citation

If you use BBB-TransAI in your research, please cite:

Kartic, Sharma A, Yi S, Park TS. Blood–brain barrier–transport artificial intelligence: A hybrid sequence–structure machine-learning framework for predicting blood–brain barrier-penetrant peptides. *Brain Network Disorders*. 2026.

## License

This project is available for academic and research purposes.

