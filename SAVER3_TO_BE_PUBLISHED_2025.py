import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio import Entrez, SeqIO
import time

# Set your email for NCBI access
Entrez.email = "dellelhomani@gmail.com"  # My NCBI email

class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = w  # real part
        self.x = x  # i component
        self.y = y  # j component
        self.z = z  # k component
    
    def __add__(self, other):
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        n = self.norm()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

class QuaternionDNA:
    def __init__(self):
        self.nuc_to_quat = {
            'A': Quaternion(1, 0, 0, 0),
            'T': Quaternion(0, 1, 0, 0),
            'G': Quaternion(0, 0, 1, 0),
            'C': Quaternion(0, 0, 0, 1),
            'N': Quaternion(0, 0, 0, 0)
        }
        
    def sequence_to_quaternions(self, sequence):
        return [self.nuc_to_quat.get(nuc, Quaternion(0, 0, 0, 0)) for nuc in sequence.upper()]
    
    def additive_walk(self, quaternions):
        walk = [quaternions[0]]
        for q in quaternions[1:]:
            walk.append(walk[-1] + q)
        return walk
    
    def get_3d_coordinates(self, quaternion_walk):
        return np.array([[q.x, q.y, q.z] for q in quaternion_walk])
    
    def plot_comparison(self, coords_wt, coords_mut, gene_name, mutation_info):
        """Create side-by-side comparison of walks and sphere projections with Start/End shown in legends."""
        fig = plt.figure(figsize=(20, 10))
        
        # ------------------------
        # Plot 1: Quaternion Walk
        # ------------------------
        ax1 = fig.add_subplot(121, projection='3d')
        line_wt, = ax1.plot(coords_wt[:, 0], coords_wt[:, 1], coords_wt[:, 2],
                            linewidth=1.5, alpha=0.9, label='Wild-type')
        line_mut, = ax1.plot(coords_mut[:, 0], coords_mut[:, 1], coords_mut[:, 2],
                             linewidth=1.5, alpha=0.9, label='Mutant')
        
        # Start/End markers (legend entries only; no text annotations)
        wt_start = ax1.scatter(coords_wt[0, 0], coords_wt[0, 1], coords_wt[0, 2], s=60, marker='o', label='WT Start')
        wt_end   = ax1.scatter(coords_wt[-1, 0], coords_wt[-1, 1], coords_wt[-1, 2], s=90, marker='*', label='WT End')
        mut_start= ax1.scatter(coords_mut[0, 0], coords_mut[0, 1], coords_mut[0, 2], s=60, marker='o', label='Mut Start')
        mut_end  = ax1.scatter(coords_mut[-1, 0], coords_mut[-1, 1], coords_mut[-1, 2], s=90, marker='*', label='Mut End')
        
        ax1.set_xlabel('i-axis (T)')
        ax1.set_ylabel('j-axis (G)')
        ax1.set_zlabel('k-axis (C)')
        ax1.set_title(f'{gene_name}: Quaternion Walk Comparison\n{mutation_info}', pad=20)
        # Legend for subplot 1
        handles1 = [line_wt, line_mut, wt_start, wt_end, mut_start, mut_end]
        ax1.legend(handles=handles1, loc='best')
        ax1.set_box_aspect([1, 1, 1])
        
        # -----------------------------
        # Plot 2: Sphere Projection
        # -----------------------------
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Normalize to unit sphere (avoid division by zero)
        eps = 1e-12
        wt_norms = np.linalg.norm(coords_wt, axis=1) + eps
        mut_norms = np.linalg.norm(coords_mut, axis=1) + eps
        norm_wt = coords_wt / wt_norms[:, np.newaxis]
        norm_mut = coords_mut / mut_norms[:, np.newaxis]
        
        line_wt_s, = ax2.plot(norm_wt[:, 0], norm_wt[:, 1], norm_wt[:, 2], linewidth=1.5, alpha=0.9, label='Wild-type')
        line_mut_s, = ax2.plot(norm_mut[:, 0], norm_mut[:, 1], norm_mut[:, 2], linewidth=1.5, alpha=0.9, label='Mutant')
        
        # Start/End markers on sphere
        wt_start_s = ax2.scatter(norm_wt[0, 0], norm_wt[0, 1], norm_wt[0, 2], s=60, marker='o', label='WT Start')
        wt_end_s   = ax2.scatter(norm_wt[-1, 0], norm_wt[-1, 1], norm_wt[-1, 2], s=90, marker='*', label='WT End')
        mut_start_s= ax2.scatter(norm_mut[0, 0], norm_mut[0, 1], norm_mut[0, 2], s=60, marker='o', label='Mut Start')
        mut_end_s  = ax2.scatter(norm_mut[-1, 0], norm_mut[-1, 1], norm_mut[-1, 2], s=90, marker='*', label='Mut End')
        
        # Optional sphere mesh for context
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax2.plot_surface(x, y, z, alpha=0.1)
        
        ax2.set_xlabel('i-axis (T)')
        ax2.set_ylabel('j-axis (G)')
        ax2.set_zlabel('k-axis (C)')
        ax2.set_title(f'{gene_name}: Sphere Projection Comparison\n{mutation_info}', pad=20)
        # Legend for subplot 2
        handles2 = [line_wt_s, line_mut_s, wt_start_s, wt_end_s, mut_start_s, mut_end_s]
        ax2.legend(handles=handles2, loc='best')
        ax2.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        return fig

def fetch_ncbi_sequence(accession, start=None, end=None):
    """Fetch DNA sequence from NCBI by accession number"""
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        seq = str(record.seq)
        if start is not None and end is not None:
            seq = seq[start:end]
        return seq
    except Exception as e:
        print(f"Error fetching sequence {accession}: {str(e)}")
        return None

def analyze_gene_comparison(gene_info):
    """Analyze and compare wild-type and mutant sequences for a gene"""
    qdna = QuaternionDNA()
    gene_name = gene_info['name']
    accession_wt = gene_info['wt_accession']
    accession_mut = gene_info['mut_accession']
    mutation_info = gene_info['mutation_info']
    start = gene_info.get('start', 0)
    end = gene_info.get('end', 500)
    
    print(f"\nAnalyzing {gene_name}...")
    
    # Fetch sequences with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            seq_wt = fetch_ncbi_sequence(accession_wt, start, end)
            seq_mut = fetch_ncbi_sequence(accession_mut, start, end)
            
            if not seq_wt or not seq_mut:
                print(f"Attempt {attempt + 1} failed - retrying...")
                time.sleep(2)  # Wait before retrying
                continue
                
            break
        except Exception as e:
            print(f"Error during attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to fetch sequences for {gene_name} after {max_retries} attempts")
                return None
            time.sleep(2)
    
    # Convert to quaternions
    quats_wt = qdna.sequence_to_quaternions(seq_wt)
    quats_mut = qdna.sequence_to_quaternions(seq_mut)
    
    # Generate additive walks
    walk_wt = qdna.additive_walk(quats_wt)
    walk_mut = qdna.additive_walk(quats_mut)
    
    # Get 3D coordinates
    coords_wt = qdna.get_3d_coordinates(walk_wt)
    coords_mut = qdna.get_3d_coordinates(walk_mut)
    
    # Create comparison plot
    fig = qdna.plot_comparison(coords_wt, coords_mut, gene_name, mutation_info)
    
    return fig

# Gene information database
genes_to_analyze = [
    {
        'name': 'CFTR',
        'wt_accession': 'NM_000492',  # CFTR wild type
        'mut_accession': 'NG_016465',  # CFTR variant
        'mutation_info': 'Mutations cause cystic fibrosis',
        'start': 0,
        'end': 500
    },
    {
        'name': 'BRCA1',
        'wt_accession': 'NM_007294',  # BRCA1 wild type
        'mut_accession': 'NM_007300',  # BRCA1 variant
        'mutation_info': 'Mutations increase breast/ovarian cancer risk',
        'start': 0,
        'end': 500
    },
    {
        'name': 'BRCA2',
        'wt_accession': 'NM_000059',  # BRCA2 wild type
        'mut_accession': 'NG_012772',  # BRCA2 variant
        'mutation_info': 'Mutations increase breast/ovarian cancer risk',
        'start': 0,
        'end': 500
    },
    {
        'name': 'HBB',
        'wt_accession': 'NM_000518',  # HBB wild type
        'mut_accession': 'NG_059281',  # HBB variant
        'mutation_info': 'Mutations cause sickle cell disease',
        'start': 0,
        'end': 500
    },
    {
        'name': 'TP53',
        'wt_accession': 'NM_000546',  # TP53 wild type
        'mut_accession': 'NG_017013',  # TP53 variant
        'mutation_info': 'Tumor suppressor mutated in many cancers',
        'start': 0,
        'end': 500
    },
    {
        'name': 'EGFR',
        'wt_accession': 'NM_005228',  # EGFR wild type
        'mut_accession': 'NG_007726',  # EGFR variant
        'mutation_info': 'Mutations associated with lung cancer',
        'start': 0,
        'end': 500
    },
    {
        'name': 'APOE',
        'wt_accession': 'NM_000041',  # APOE wild type
        'mut_accession': 'NG_007084',  # APOE variant
        'mutation_info': 'Variations linked to Alzheimer\'s disease',
        'start': 0,
        'end': 500
    },
    {
        'name': 'FBN1',
        'wt_accession': 'NM_000138',  # FBN1 wild type
        'mut_accession': 'NG_008805',  # FBN1 variant
        'mutation_info': 'Mutations cause Marfan syndrome',
        'start': 0,
        'end': 500
    },
    {
        'name': 'HTT',
        'wt_accession': 'NM_002111',  # HTT wild type
        'mut_accession': 'NG_009378',  # HTT variant
        'mutation_info': 'Mutations cause Huntington\'s disease',
        'start': 0,
        'end': 500
    },
    {
        'name': 'DMD',
        'wt_accession': 'NM_004006',  # DMD wild type
        'mut_accession': 'NG_012232',  # DMD variant
        'mutation_info': 'Mutations cause Duchenne muscular dystrophy',
        'start': 0,
        'end': 500
    },
    {
        'name': 'PKD1',
        'wt_accession': 'NM_000296',  # PKD1 wild type
        'mut_accession': 'NG_008617',  # PKD1 variant
        'mutation_info': 'Mutations cause polycystic kidney disease',
        'start': 0,
        'end': 500
    }
    # I Have removed PKD2 to be honest I couldn't retrieve it from database!!!!!
]

# Main analysis
if __name__ == "__main__":
    for gene_info in genes_to_analyze:
        fig = analyze_gene_comparison(gene_info)
        if fig:
            plt.show()
        time.sleep(1)  # Be kind to NCBI servers
