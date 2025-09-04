from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Load all SDF files
sdf_files = glob.glob("*.sdf")  # adjust path if needed
mols = {}
for sdf in sdf_files:
    suppl = Chem.SDMolSupplier(sdf)
    mol = suppl[0]  # first molecule in file
    if mol:
        name = sdf.split(".")[0]
        mols[name] = mol

# Ensure native MZB is present
if "MZB" not in mols:
    raise ValueError("Native MZB (MZB.sdf) not found in the folder!")

# Generate FP2 fingerprints (path-based)
fps = {name: Chem.RDKFingerprint(mol) for name, mol in mols.items()}

# Compute similarity matrix (FP2 Tanimoto)
names = list(fps.keys())
n = len(names)
sim_matrix = pd.DataFrame(0.0, index=names, columns=names)

for i in range(n):
    for j in range(n):
        sim = DataStructs.TanimotoSimilarity(fps[names[i]], fps[names[j]])
        sim_matrix.iloc[i, j] = sim

# Clustered heatmap
sns.clustermap(
    sim_matrix,
    cmap="viridis",
    figsize=(12, 12),
    annot=False,
    cbar_kws={"label": "Tanimoto Similarity (FP2)"}
)

plt.suptitle("Clustered FP2 Similarity Heatmap of MZB and Its Modifications", y=1.02, fontsize=14)
plt.show()