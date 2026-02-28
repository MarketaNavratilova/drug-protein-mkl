# Multiple Kernel Learning for Drug-Protein Binding Affinity Prediction

## Overview

A Python implementation of a **Multiple Kernel Learning (MKL)** regression framework for predicting drug-protein binding affinities (pKi). This project translates my previous academic work (originally in R) into a scalable Python workflow.

The goal is to predict how strongly a drug molecule binds to a target protein by combining multiple representations of chemical structures and amino acid sequences.

## Methodology

The pipeline follows a pairwise learning approach, assuming that similar drugs interact with similar proteins.

### Data Representation

- **Drugs**: Represented as molecular fingerprints (Morgan/Circular, MACCS keys)
- **Proteins**: Represented as k-mer spectrum vectors from amino acid sequences
- **Binding Affinity**: Measured as pKi (negative log of the inhibition constant)

### Kernel Construction

Multiple kernels capture complementary aspects of drug and protein similarity:

- **Drug Kernels**: Tanimoto similarity on Morgan fingerprints (radius 2 and 3) and MACCS structural keys — 3 drug kernels total
- **Protein Kernels**: Normalized k-mer spectrum kernels (k=3 and k=4), computing cosine similarity in k-mer count space — 2 protein kernels total
- **Pairwise Kernels**: Each (drug kernel, protein kernel) pair is combined via Kronecker product, yielding 6 pairwise kernels

### Machine Learning

- **Framework**: Multiple Kernel Learning (UNIMKL — uniform weighting of all pairwise kernels)
- **Algorithm**: Kernel Ridge Regression (KRR) with precomputed combined kernel
- **Hyperparameter Tuning**: Cross-validated regularization parameter (alpha)
- **Evaluation**: Drug-level GroupKFold cross-validation (Setting S2: test drugs are never seen during training), reporting RMSE and Pearson correlation with standard deviations

## Project Structure

```
├── main.py                  # Core pipeline: kernels, MKL, training, evaluation
├── generate_dummy_data.py   # Synthetic data with biologically meaningful signal
├── requirements.txt         # Python dependencies
└── data/                    # Generated CSV files
    ├── drugs.csv            # Drug IDs and SMILES strings
    ├── proteins.csv         # Protein IDs and amino acid sequences
    └── interactions.csv     # Measured pKi values (sparse)
```

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate sample data** (if you don't have biological data ready):
   ```bash
   python generate_dummy_data.py
   ```

3. **Run the model**:
   ```bash
   python main.py
   ```

## Design Decisions

**Why multiple kernels?** Different molecular representations capture different aspects of chemical similarity. Morgan fingerprints encode local circular environments while MACCS keys capture predefined structural patterns. Combining them via MKL lets the model leverage complementary information.

**Why drug-level cross-validation?** Random pair-level splits allow the model to see the same drug in both training and testing, leading to overly optimistic performance. Drug-level splits (Setting S2) evaluate the model's ability to generalize to entirely new compounds, which is the realistic use case in drug discovery.

**Why k-mer spectrum kernels for proteins?** Unlike position-wise matching, k-mer spectrum kernels are valid positive semi-definite kernels that capture local sequence composition regardless of alignment or sequence length. They are a principled approximation to more expensive string kernels.

## Dependencies

- Python 3.8+
- NumPy, Pandas, SciPy
- Scikit-learn
- RDKit (for molecular fingerprint generation)
