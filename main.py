"""
Multiple Kernel Learning (MKL) for Drug-Protein Binding Affinity Prediction.

Pipeline:
    1. Load drug, protein, and interaction data
    2. Compute multiple drug kernels (Morgan FP at different radii, MACCS keys)
    3. Compute multiple protein kernels (k-mer spectrum at different k)
    4. Construct pairwise kernels via Kronecker product
    5. Combine with uniform weights (UNIMKL)
    6. Train Kernel Ridge Regression with cross-validated alpha
    7. Evaluate with drug-level split (Setting S2: new drugs at test time)
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import product as iterproduct

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit import DataStructs

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load and validate drug, protein, and interaction data."""
    print("1. Loading data...")
    try:
        drugs = pd.read_csv('data/drugs.csv').sort_values('drug_id').reset_index(drop=True)
        proteins = pd.read_csv('data/proteins.csv').sort_values('protein_id').reset_index(drop=True)
        interactions = pd.read_csv('data/interactions.csv')
    except FileNotFoundError:
        print("   Error: Data files not found. Run 'python generate_dummy_data.py' first.")
        exit(1)

    drug_ids = drugs['drug_id'].tolist()
    protein_ids = proteins['protein_id'].tolist()

    # Filter interactions to only known drugs/proteins
    interactions = interactions[
        interactions['drug_id'].isin(drug_ids) &
        interactions['protein_id'].isin(protein_ids)
    ].copy()

    print(f"   {len(drugs)} drugs, {len(proteins)} proteins, {len(interactions)} measured pairs")
    return drugs, proteins, interactions


# ═══════════════════════════════════════════════════════════════════════
# DRUG KERNELS
# ═══════════════════════════════════════════════════════════════════════

def tanimoto_kernel_from_fps(fps):
    """Compute Tanimoto similarity matrix from RDKit fingerprints."""
    n = len(fps)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            K[i, j] = sim
            K[j, i] = sim
    return K


def validate_smiles(smiles_list):
    """Parse SMILES and return molecules, flagging any that fail."""
    mols = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            print(f"   WARNING: Invalid SMILES at index {i}: '{s}' — skipping")
        mols.append(mol)
    return mols


def compute_morgan_kernel(smiles_list, radius=2, n_bits=2048, mols=None):
    """Drug kernel from Morgan (circular) fingerprints with Tanimoto similarity."""
    if mols is None:
        mols = validate_smiles(smiles_list)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprint(m) for m in mols]
    return tanimoto_kernel_from_fps(fps)


def compute_maccs_kernel(smiles_list, mols=None):
    """Drug kernel from MACCS structural keys with Tanimoto similarity."""
    if mols is None:
        mols = validate_smiles(smiles_list)
    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    return tanimoto_kernel_from_fps(fps)


def compute_drug_kernels(smiles_list):
    """Compute multiple drug kernels for MKL. Validates SMILES first."""
    print("2. Computing drug kernels...")

    # Validate all SMILES once upfront
    mols = validate_smiles(smiles_list)
    valid_mask = [m is not None for m in mols]

    if not all(valid_mask):
        n_bad = sum(1 for v in valid_mask if not v)
        print(f"   Removing {n_bad} drug(s) with invalid SMILES")
        mols = [m for m in mols if m is not None]

    kernels = {}
    kernels['morgan_r2'] = compute_morgan_kernel(smiles_list, radius=2, mols=mols)
    kernels['morgan_r3'] = compute_morgan_kernel(smiles_list, radius=3, mols=mols)
    kernels['maccs'] = compute_maccs_kernel(smiles_list, mols=mols)
    print(f"   {len(kernels)} drug kernels: {list(kernels.keys())}")
    return kernels, valid_mask


# ═══════════════════════════════════════════════════════════════════════
# PROTEIN KERNELS
# ═══════════════════════════════════════════════════════════════════════

def kmer_spectrum(sequence, k):
    """Count all k-mers in a protein sequence."""
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def compute_spectrum_kernel(sequences, k=3):
    """
    Protein kernel using the k-mer spectrum (bag of k-mers).

    This is a valid Mercer kernel: the inner product of k-mer count vectors,
    normalized to unit length (cosine similarity in k-mer space).
    """
    spectra = [kmer_spectrum(seq, k) for seq in sequences]

    # Gather all k-mers across sequences
    all_kmers = sorted(set().union(*spectra))
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}

    # Build count vectors
    n = len(sequences)
    dim = len(all_kmers)
    X = np.zeros((n, dim))
    for i, spec in enumerate(spectra):
        for km, count in spec.items():
            X[i, kmer_to_idx[km]] = count

    # Normalize rows to unit length (cosine kernel)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    return X_norm @ X_norm.T


def compute_protein_kernels(sequences):
    """Compute multiple protein kernels for MKL."""
    print("3. Computing protein kernels...")
    kernels = {}
    kernels['spectrum_k3'] = compute_spectrum_kernel(sequences, k=3)
    kernels['spectrum_k4'] = compute_spectrum_kernel(sequences, k=4)
    print(f"   {len(kernels)} protein kernels: {list(kernels.keys())}")
    return kernels


# ═══════════════════════════════════════════════════════════════════════
# PAIRWISE KERNEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_pairwise_kernels(drug_kernels, protein_kernels):
    """
    Construct pairwise kernels via Kronecker product.

    For each combination of drug kernel and protein kernel:
        K_pair = K_drug ⊗ K_protein

    This yields one pairwise kernel per (drug_kernel, protein_kernel) pair.
    """
    print("4. Constructing pairwise kernels (Kronecker product)...")
    pairwise_kernels = {}
    for d_name, K_d in drug_kernels.items():
        for p_name, K_p in protein_kernels.items():
            name = f"{d_name}_x_{p_name}"
            pairwise_kernels[name] = np.kron(K_d, K_p)

    print(f"   {len(pairwise_kernels)} pairwise kernels: {list(pairwise_kernels.keys())}")
    first_key = next(iter(pairwise_kernels))
    print(f"   Kernel shape: {pairwise_kernels[first_key].shape}")
    return pairwise_kernels


def unimkl_combine(pairwise_kernels):
    """Uniform MKL: average all pairwise kernels with equal weights."""
    kernels_list = list(pairwise_kernels.values())
    M = len(kernels_list)
    K_combined = sum(kernels_list) / M
    return K_combined


# ═══════════════════════════════════════════════════════════════════════
# TARGET VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_target_vector(drugs, proteins, interactions):
    """
    Build target vector and mask for observed drug-protein pairs.

    The Kronecker kernel is indexed as (drug_i * n_proteins + protein_j),
    so y must follow the same ordering.

    Returns:
        y_full:   target values for ALL pairs (NaN for unobserved)
        observed: boolean mask of which pairs have measurements
        drug_idx: drug index for each pair (for group-based CV)
    """
    drug_ids = drugs['drug_id'].tolist()
    protein_ids = proteins['protein_id'].tolist()
    n_drugs = len(drug_ids)
    n_proteins = len(protein_ids)

    # Create lookup for fast access
    interaction_map = {}
    for _, row in interactions.iterrows():
        interaction_map[(row['drug_id'], row['protein_id'])] = row['pKi']

    y_full = np.full(n_drugs * n_proteins, np.nan)
    drug_idx = np.zeros(n_drugs * n_proteins, dtype=int)

    for i, d_id in enumerate(drug_ids):
        for j, p_id in enumerate(protein_ids):
            flat_idx = i * n_proteins + j
            drug_idx[flat_idx] = i
            if (d_id, p_id) in interaction_map:
                y_full[flat_idx] = interaction_map[(d_id, p_id)]

    observed = ~np.isnan(y_full)
    return y_full, observed, drug_idx


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_with_drug_split(K_combined, y_full, observed, drug_idx, drugs):
    """
    Evaluate using drug-level GroupKFold (Setting S2: new drugs at test time).

    This ensures that all pairs involving a given drug are either entirely
    in the training set or entirely in the test set, preventing data leakage
    through shared drug information.
    """
    print("5. Evaluating with drug-level cross-validation (Setting S2)...")

    # Work only with observed pairs
    obs_indices = np.where(observed)[0]
    y_obs = y_full[obs_indices]
    drug_groups = drug_idx[obs_indices]

    n_unique_drugs = len(np.unique(drug_groups))
    n_splits = min(5, n_unique_drugs)

    if n_splits < 2:
        print("   Warning: Not enough drugs for cross-validation.")
        return

    gkf = GroupKFold(n_splits=n_splits)

    # Cross-validated alpha selection
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_alpha = alphas[0]
    best_rmse = np.inf

    print(f"   Tuning regularization (alpha) over {n_splits} folds...")

    for alpha in alphas:
        fold_rmses = []
        for train_idx, test_idx in gkf.split(obs_indices, y_obs, groups=drug_groups):
            train_global = obs_indices[train_idx]
            test_global = obs_indices[test_idx]

            K_tr = K_combined[np.ix_(train_global, train_global)]
            K_te = K_combined[np.ix_(test_global, train_global)]

            krr = KernelRidge(alpha=alpha, kernel='precomputed')
            krr.fit(K_tr, y_obs[train_idx])
            y_pred = krr.predict(K_te)

            rmse = np.sqrt(mean_squared_error(y_obs[test_idx], y_pred))
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_alpha = alpha

    print(f"   Best alpha: {best_alpha} (CV RMSE: {best_rmse:.4f})")

    # Final evaluation with best alpha
    print(f"\n   Final {n_splits}-fold evaluation (alpha={best_alpha}):")
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(
        gkf.split(obs_indices, y_obs, groups=drug_groups)
    ):
        train_global = obs_indices[train_idx]
        test_global = obs_indices[test_idx]

        K_tr = K_combined[np.ix_(train_global, train_global)]
        K_te = K_combined[np.ix_(test_global, train_global)]

        krr = KernelRidge(alpha=best_alpha, kernel='precomputed')
        krr.fit(K_tr, y_obs[train_idx])
        y_pred = krr.predict(K_te)

        rmse = np.sqrt(mean_squared_error(y_obs[test_idx], y_pred))
        corr, _ = pearsonr(y_obs[test_idx], y_pred)
        fold_results.append({'fold': fold_i + 1, 'rmse': rmse, 'pearson': corr,
                             'n_test': len(test_idx)})
        print(f"   Fold {fold_i+1}: RMSE={rmse:.4f}, Pearson={corr:.4f} "
              f"({len(test_idx)} test pairs)")

    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    mean_corr = np.mean([r['pearson'] for r in fold_results])
    std_corr = np.std([r['pearson'] for r in fold_results])

    return mean_rmse, std_rmse, mean_corr, std_corr, best_alpha


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline():
    drugs, proteins, interactions = load_data()

    # Compute drug kernels (with SMILES validation)
    drug_kernels, valid_mask = compute_drug_kernels(drugs['smiles'].tolist())

    # Filter out drugs with invalid SMILES
    if not all(valid_mask):
        valid_drug_ids = drugs.loc[valid_mask, 'drug_id'].tolist()
        drugs = drugs[valid_mask].reset_index(drop=True)
        interactions = interactions[interactions['drug_id'].isin(valid_drug_ids)].copy()
        print(f"   After filtering: {len(drugs)} valid drugs, {len(interactions)} pairs")

    protein_kernels = compute_protein_kernels(proteins['sequence'].tolist())

    # Construct and combine pairwise kernels (UNIMKL)
    pairwise_kernels = build_pairwise_kernels(drug_kernels, protein_kernels)
    K_combined = unimkl_combine(pairwise_kernels)

    # Build target vector (handles sparse data)
    y_full, observed, drug_idx = build_target_vector(drugs, proteins, interactions)
    n_obs = observed.sum()
    n_total = len(y_full)
    print(f"   {n_obs}/{n_total} observed pairs ({n_obs/n_total*100:.0f}% coverage)")

    # Evaluate with proper drug-level split
    results = evaluate_with_drug_split(K_combined, y_full, observed, drug_idx, drugs)

    if results is not None:
        mean_rmse, std_rmse, mean_corr, std_corr, best_alpha = results

        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)
        print(f"  Method:       UNIMKL + Kernel Ridge Regression")
        print(f"  Drug kernels: Morgan (r=2), Morgan (r=3), MACCS")
        print(f"  Prot kernels: 3-mer spectrum, 4-mer spectrum")
        print(f"  Pairwise:     Kronecker product ({len(pairwise_kernels)} combinations)")
        print(f"  Evaluation:   Drug-level GroupKFold (Setting S2)")
        print(f"  Alpha:        {best_alpha}")
        print(f"  RMSE:         {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  Pearson:      {mean_corr:.4f} ± {std_corr:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
