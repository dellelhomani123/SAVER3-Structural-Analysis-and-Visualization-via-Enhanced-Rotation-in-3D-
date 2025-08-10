import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from Bio import Entrez, SeqIO
import pandas as pd

# Set your email for NCBI
Entrez.email = "dellelhomani@gmail.com"  # My NCBI email 

# Quaternion class
class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        n = self.norm()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n) if n != 0 else Quaternion(0, 0, 0, 0)
    def vector_part(self):
        return np.array([self.x, self.y, self.z])

# Mapping and Walk
def nucleotide_to_quaternion(nt):
    return {
        'A': Quaternion(1, 0, 0, 0),
        'T': Quaternion(0, 1, 0, 0),
        'G': Quaternion(0, 0, 1, 0),
        'C': Quaternion(0, 0, 0, 1)
    }.get(nt.upper(), Quaternion(0, 0, 0, 0))

def quaternion_walk(seq):
    walk = []
    current = Quaternion(0, 0, 0, 0)
    for nt in seq:
        current += nucleotide_to_quaternion(nt)
        walk.append(current)
    return walk

# Metrics
def compute_cosine_distance(wt, mut):
    wt_vecs = np.array([q.vector_part() for q in wt])
    mut_vecs = np.array([q.vector_part() for q in mut])
    min_len = min(len(wt_vecs), len(mut_vecs))
    return cosine(wt_vecs[:min_len].flatten(), mut_vecs[:min_len].flatten())

def compute_entropy(seq):
    values, counts = np.unique(list(seq), return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def fetch_genbank_sequence(accession, start=0, end=500):
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    return str(record.seq)[start:end]

def safe_normalize(coords):
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return coords / norms

# Full gene list
gene_accessions = {
    "BRCA1": ("NM_007294", "NM_007300"),
    "BRCA2": ("NM_000059", "NG_012772"),  # Added BRCA2 here
    "CFTR":  ("NM_000492", "NG_016465"),
    "HBB":   ("NM_000518", "NG_059281"),
    "TP53":  ("NM_000546", "NG_017013"),
    "EGFR":  ("NM_005228", "NG_007726"),
    "APOE":  ("NM_000041", "NG_007084"),
    "FBN1":  ("NM_000138", "NG_008805"),
    "HTT":   ("NM_002111", "NG_009378"),
    "DMD":   ("NM_004006", "NG_012232"),
    "PKD1":  ("NM_000296", "NG_008617")
    # Removed PKD2 and added BRCA2
}


metrics = []

# Process all genes
for gene, (acc_wt, acc_mut) in gene_accessions.items():
    print(f"Processing {gene}...")

    wt_seq = fetch_genbank_sequence(acc_wt)
    mut_seq = fetch_genbank_sequence(acc_mut)

    wt_walk = quaternion_walk(wt_seq)
    mut_walk = quaternion_walk(mut_seq)

    cos_dist = compute_cosine_distance(wt_walk, mut_walk)
    ent_wt = compute_entropy(wt_seq)
    ent_mut = compute_entropy(mut_seq)

    wt_coords = np.array([q.vector_part() for q in wt_walk])
    mut_coords = np.array([q.vector_part() for q in mut_walk])

    wt_norm = safe_normalize(wt_coords)
    mut_norm = safe_normalize(mut_coords)

    # Plotting
    #fig = plt.figure(figsize=(12, 5))
    #ax1 = fig.add_subplot(121, projection='3d')
    #ax1.plot(wt_coords[:, 0], wt_coords[:, 1], wt_coords[:, 2], label='WT', color='blue')
    #ax1.plot(mut_coords[:, 0], mut_coords[:, 1], mut_coords[:, 2], label='Mutant', color='red')
    #ax1.set_title(f'{gene} Quaternion Walk')
    #ax1.legend()

    #ax2 = fig.add_subplot(122, projection='3d')
    #ax2.scatter(wt_norm[:, 0], wt_norm[:, 1], wt_norm[:, 2], color='blue', s=2, label='WT')
    #ax2.scatter(mut_norm[:, 0], mut_norm[:, 1], mut_norm[:, 2], color='red', s=2, label='Mutant')
    #ax2.set_title(f'{gene} Quaternion Sphere')
    #ax2.legend()
    #plt.suptitle(f"{gene} - Quaternion DNA Representation", fontsize=14)
    #plt.tight_layout()
    #plt.show()

    metrics.append([gene, round(cos_dist, 4), round(ent_wt, 4), round(ent_mut, 4)])

# Display final table
df = pd.DataFrame(metrics, columns=["Gene", "Cosine Distance", "Entropy (WT)", "Entropy (Mutant)"])
print("\n=== Quaternion DNA Analysis Metrics ===")
print(df.to_string(index=False))
