import numpy as np
import matplotlib.pyplot as plt
from Bio import Entrez, SeqIO
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import pandas as pd

# Set your email for NCBI API access
Entrez.email = "dellelhomani@gmail.com"  # <-- My NCBI Email

# List of genes with (WT, Mutant) GenBank accession numbers
gene_accessions = {
    "BRCA1": ("NM_007294", "NM_007300"),
    "BRCA2": ("NM_000059", "NG_012772"),  # Added BRCA2
    "CFTR":  ("NM_000492", "NG_016465"),
    "HBB":   ("NM_000518", "NG_059281"),
    "TP53":  ("NM_000546", "NG_017013"),
    "EGFR":  ("NM_005228", "NG_007726"),
    "APOE":  ("NM_000041", "NG_007084"),
    "FBN1":  ("NM_000138", "NG_008805"),
    "HTT":   ("NM_002111", "NG_009378"),
    "DMD":   ("NM_004006", "NG_012232"),
    "PKD1":  ("NM_000296", "NG_008617")
    # Removed PKD2
}

# Fetch GenBank sequence from NCBI
def fetch_genbank_sequence(accession, start=0, end=500):
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    return str(record.seq)[start:end]

# Generate Chaos Game Representation matrix
def generate_cgr(sequence, size=256):
    corners = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}
    img = np.zeros((size, size))
    x, y = 0.5, 0.5
    for nt in sequence.upper():
        if nt not in corners:
            continue
        cx, cy = corners[nt]
        x = (x + cx) / 2
        y = (y + cy) / 2
        img[int((1 - y) * (size - 1)), int(x * (size - 1))] += 1
    return img

# Generate Z-curve 3D trajectory
def generate_z_curve(sequence):
    x, y, z = [0], [0], [0]
    for nt in sequence.upper():
        dx = 1 if nt in "AG" else -1
        dy = 1 if nt in "AC" else -1
        dz = 1 if nt in "AT" else -1
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        z.append(z[-1] + dz)
    return np.array(x), np.array(y), np.array(z)

# Quaternion class for SAVER3 method
class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    def vector_part(self):
        return np.array([self.x, self.y, self.z])

# Map nucleotide to quaternion
def nucleotide_to_quaternion(nt):
    return {
        'A': Quaternion(1, 0, 0, 0),
        'T': Quaternion(0, 1, 0, 0),
        'G': Quaternion(0, 0, 1, 0),
        'C': Quaternion(0, 0, 0, 1)
    }.get(nt.upper(), Quaternion(0, 0, 0, 0))

# Build quaternion walk
def quaternion_walk(seq):
    walk = []
    current = Quaternion(0, 0, 0, 0)
    for nt in seq:
        current += nucleotide_to_quaternion(nt)
        walk.append(current)
    return np.array([q.vector_part() for q in walk])

# Initialize comparison table
metrics_comparison = []

# Process all genes
for gene, (acc_wt, acc_mut) in gene_accessions.items():
    print(f"Processing {gene}...")

    # Fetch WT and Mutant sequences
    seq_wt = fetch_genbank_sequence(acc_wt)
    seq_mut = fetch_genbank_sequence(acc_mut)

    # Entropy
    ent_wt = entropy(pd.Series(list(seq_wt)).value_counts(normalize=True), base=2)
    ent_mut = entropy(pd.Series(list(seq_mut)).value_counts(normalize=True), base=2)

    # CGR Cosine Distance
    cgr_wt = generate_cgr(seq_wt)
    cgr_mut = generate_cgr(seq_mut)
    cgr_cosine = cosine(cgr_wt.flatten(), cgr_mut.flatten())

    # Z-Curve Cosine Distance
    z_wt = np.array(generate_z_curve(seq_wt)).T
    z_mut = np.array(generate_z_curve(seq_mut)).T
    min_len = min(len(z_wt), len(z_mut))
    z_cosine = cosine(z_wt[:min_len].flatten(), z_mut[:min_len].flatten())

    # Quaternion Walk Cosine Distance
    qw_wt = quaternion_walk(seq_wt)
    qw_mut = quaternion_walk(seq_mut)
    qw_cosine = cosine(qw_wt[:min_len].flatten(), qw_mut[:min_len].flatten())

    # Store metrics
    metrics_comparison.append([
        gene,
        round(ent_wt, 4), round(ent_mut, 4),
        round(cgr_cosine, 4),
        round(z_cosine, 4),
        round(qw_cosine, 4)
    ])

# Create DataFrame and display
df_comparison = pd.DataFrame(metrics_comparison, columns=[
    "Gene", "Entropy WT", "Entropy Mutant",
    "CGR Cosine", "Z-Curve Cosine", "Quaternion Cosine"
])

print("\n=== Multi-Method Genomic Comparison ===\n")
print(df_comparison.to_string(index=False))
