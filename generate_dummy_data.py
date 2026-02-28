"""
Generate synthetic drug-protein binding affinity data for demonstration.

Creates realistic dummy data where the planted signal correlates with
actual molecular/sequence features so that the kernels can learn it:
- Aromatic/hydrophobic drugs bind more strongly to hydrophobic-rich proteins
- Data is sparse (~40% of pairs measured) to mimic real-world conditions
"""

import pandas as pd
import numpy as np
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if not os.path.exists('data'):
    os.makedirs('data')


def compute_aromaticity_score(smiles: str) -> float:
    """Heuristic aromaticity: fraction of lowercase chars in SMILES (aromatic atoms)."""
    aromatic_chars = sum(1 for c in smiles if c.islower() and c.isalpha())
    total_chars = sum(1 for c in smiles if c.isalpha())
    return aromatic_chars / total_chars if total_chars > 0 else 0.0


def compute_hydrophobicity_score(sequence: str) -> float:
    """Fraction of hydrophobic residues in a protein sequence."""
    hydrophobic = set("AILMFWVP")
    return sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)


def create_dummy_data():
    print("Generating synthetic biological data...")
    print(f"Random seed: {SEED}")

    # ── 1. Drug compounds (real SMILES) ──────────────────────────────
    drug_entries = [
        ("CC(=O)Oc1ccccc1C(=O)O",         "Aspirin"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",    "Ibuprofen"),
        ("CC(=O)Nc1ccc(O)cc1",             "Paracetamol"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  "Caffeine"),
        ("c1ccc2ccccc2c1",                 "Naphthalene"),
        ("c1ccc2cc3ccccc3cc2c1",          "Anthracene"),
        ("c1ccncc1",                       "Pyridine"),
        ("c1cc[nH]c1",                     "Pyrrole"),
        ("c1ccoc1",                        "Furan"),
        ("c1ccsc1",                        "Thiophene"),
        ("c1c[nH]cn1",                     "Imidazole"),
        ("c1cncnc1",                       "Pyrimidine"),
        ("OC(=O)CCCCC",                    "Hexanoic acid"),
        ("CCCCCCCCCC",                     "Decane"),
        ("OC(=O)CC(O)(CC(=O)O)C(=O)O",    "Citric acid"),
        ("OCC(O)CO",                       "Glycerol"),
        ("CC(O)=O",                        "Acetic acid"),
        ("NCCO",                           "Ethanolamine"),
        ("OC(=O)/C=C/C(=O)O",             "Fumaric acid"),
        ("OC(=O)C(O)C(O)C(=O)O",          "Tartaric acid"),
    ]

    drugs = [
        {"drug_id": f"D{i:03d}", "smiles": smiles, "name": name}
        for i, (smiles, name) in enumerate(drug_entries)
    ]
    df_drugs = pd.DataFrame(drugs)
    df_drugs.to_csv('data/drugs.csv', index=False)
    print(f"  Created data/drugs.csv ({len(drugs)} compounds)")

    # ── 2. Protein sequences ─────────────────────────────────────────
    # Two groups: hydrophobic-rich and polar-rich proteins
    hydrophobic_aa = "AILMFWVP"
    polar_aa = "DEKRNQHSTY"
    all_aa = "ACDEFGHIKLMNPQRSTVWY"
    n_proteins = 20
    proteins = []

    for i in range(n_proteins):
        length = random.randint(80, 150)
        if i < 10:
            # Hydrophobic-biased (60% hydrophobic, 40% polar)
            seq = "".join(
                random.choice(hydrophobic_aa) if random.random() < 0.6
                else random.choice(polar_aa)
                for _ in range(length)
            )
        else:
            # Polar-biased (30% hydrophobic, 70% polar)
            seq = "".join(
                random.choice(hydrophobic_aa) if random.random() < 0.3
                else random.choice(polar_aa)
                for _ in range(length)
            )
        proteins.append({"protein_id": f"P{i:03d}", "sequence": seq})

    df_proteins = pd.DataFrame(proteins)
    df_proteins.to_csv('data/proteins.csv', index=False)
    print(f"  Created data/proteins.csv ({n_proteins} targets)")

    # ── 3. Binding affinities with feature-correlated signal ─────────
    interactions = []
    n_total_possible = len(drugs) * n_proteins
    n_measured = 0

    for d in drugs:
        arom_score = compute_aromaticity_score(d["smiles"])
        for p in proteins:
            # Sparsity: randomly measure ~40% of pairs
            if random.random() > 0.40:
                continue

            hydro_score = compute_hydrophobicity_score(p["sequence"])

            # Base affinity + signal: aromatic drugs bind hydrophobic proteins more strongly
            base_affinity = np.random.normal(loc=6.0, scale=0.8)
            signal = 2.5 * arom_score * hydro_score  # up to ~1.5 pKi boost
            noise = np.random.normal(0, 0.3)
            pki = np.clip(base_affinity + signal + noise, 3.0, 10.0)

            interactions.append({
                "drug_id": d["drug_id"],
                "protein_id": p["protein_id"],
                "pKi": round(pki, 2),
            })
            n_measured += 1

    df_interactions = pd.DataFrame(interactions)
    df_interactions.to_csv('data/interactions.csv', index=False)
    sparsity = n_measured / n_total_possible * 100
    print(f"  Created data/interactions.csv ({n_measured}/{n_total_possible} pairs, {sparsity:.0f}% coverage)")
    print("Done!")


if __name__ == "__main__":
    create_dummy_data()
