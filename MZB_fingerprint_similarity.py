from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, RDKFingerprint, rdFingerprintGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ---- Load all SDF files in current folder ----
sdf_files = glob.glob("*.sdf")
mols = {}

for sdf in sdf_files:
    suppl = Chem.SDMolSupplier(sdf)
    mol = suppl[0]  # first molecule from each sdf
    if mol:
        name = os.path.splitext(os.path.basename(sdf))[0]  # file name without extension
        mols[name] = mol

# ---- Define native MZB ----
native_name = "MZB"  # <-- ensure native is named "MZB.sdf"
if native_name not in mols:
    raise ValueError("Native MZB (MZB.sdf) not found in folder!")

native = mols[native_name]

# ---- Generate fingerprints ----
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fps_morgan = {name: morgan_gen.GetFingerprint(mol) for name, mol in mols.items()}
fps_fp2 = {name: RDKFingerprint(mol) for name, mol in mols.items()}

# ---- Compute similarities to native ----
similarities = []
for name, mol in mols.items():
    if name == native_name:
        continue
    sim_morgan = DataStructs.TanimotoSimilarity(fps_morgan[native_name], fps_morgan[name])
    sim_fp2 = DataStructs.TanimotoSimilarity(fps_fp2[native_name], fps_fp2[name])
    similarities.append([name, sim_morgan, sim_fp2])

df = pd.DataFrame(similarities, columns=["Analog", "Morgan (ECFP4)", "FP2 (Path)"])

# ---- Sort by FP2 similarity ----
df_sorted = df.sort_values("FP2 (Path)", ascending=False).reset_index(drop=True)

# ---- Save table ----
df_sorted.to_csv("MZB_similarity_comparison_sorted_FP2.csv", index=False)
print(df_sorted.head())

# ---- Plot comparison ----
df_sorted.set_index("Analog")[["Morgan (ECFP4)", "FP2 (Path)"]].plot(kind="bar", figsize=(16,6))
plt.ylabel("Tanimoto Similarity to Native MZB")
plt.title("Comparison of Morgan vs FP2 Similarities of MZB Analogs to Native MZB")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()