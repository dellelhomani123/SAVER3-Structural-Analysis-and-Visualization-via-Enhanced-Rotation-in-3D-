import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import time
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# ====================== SAVER3 IMPLEMENTATION ======================
class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = w  # real part
        self.x = x  # i component
        self.y = y  # j component
        self.z = z  # k component
    
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
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

# ====================== CGR IMPLEMENTATION ======================
def chaos_game_representation(sequence, k=1):
    points = {'A': (0, 0), 'T': (1, 0), 'G': (1, 1), 'C': (0, 1)}
    cgr = np.zeros((2**k, 2**k))
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        x, y = np.mean([points[base] for base in kmer], axis=0)
        cgr[int(x*(2**k-1)), int(y*(2**k-1))] += 1
    return cgr / (cgr.sum() + 1e-10)

# ====================== Z-CURVE IMPLEMENTATION ======================
def z_curve(sequence):
    x = np.cumsum([(base == 'G') - (base == 'C') for base in sequence])
    y = np.cumsum([(base == 'A') - (base == 'T') for base in sequence])
    z = np.cumsum([(base in 'AT') - (base in 'CG') for base in sequence])
    return np.column_stack((x, y, z))

# ====================== SEQUENCE SIMULATION ======================
def simulate_sequences(num_pairs=100, length=500):
    bases = np.array(['A', 'T', 'G', 'C'])
    sequences = []
    
    for _ in range(num_pairs):
        wt = ''.join(np.random.choice(bases, size=length))
        
        mut_type = np.random.choice(['SNP', 'indel', 'structural'], p=[0.4, 0.3, 0.3])
        
        if mut_type == 'SNP':
            pos = np.random.randint(length)
            mut = wt[:pos] + np.random.choice([b for b in bases if b != wt[pos]]) + wt[pos+1:]
        elif mut_type == 'indel':
            pos = np.random.randint(length-10)
            if np.random.rand() > 0.5:
                ins_len = np.random.randint(1, 6)
                mut = wt[:pos] + ''.join(np.random.choice(bases, size=ins_len)) + wt[pos:]
            else:
                del_len = np.random.randint(1, 6)
                mut = wt[:pos] + wt[pos+del_len:]
        else:
            pos = np.random.randint(length-20)
            if np.random.rand() > 0.5:
                mut = wt[:pos] + ''.join(np.random.choice(bases, size=20)) + wt[pos:]
            else:
                mut = wt[:pos] + wt[pos+20:]
        
        sequences.append((wt, mut, mut_type))
    
    return sequences

# ====================== BENCHMARKING ======================
def run_benchmark(sequences):
    qdna = QuaternionDNA()
    results = {'SAVER3': {'cosine': [], 'entropy': [], 'time': []},
               'CGR': {'cosine': [], 'entropy': [], 'time': []},
               'Z-curve': {'cosine': [], 'entropy': [], 'time': []}}
    
    for wt, mut, mut_type in tqdm(sequences, desc="Benchmarking"):
        # SAVER3
        start = time.time()
        quats_wt = qdna.sequence_to_quaternions(wt)
        quats_mut = qdna.sequence_to_quaternions(mut)
        min_len = min(len(quats_wt), len(quats_mut))
        coords_wt = qdna.get_3d_coordinates(qdna.additive_walk(quats_wt[:min_len]))
        coords_mut = qdna.get_3d_coordinates(qdna.additive_walk(quats_mut[:min_len]))
        flat_wt = coords_wt.flatten()
        flat_mut = coords_mut.flatten()
        max_len = max(len(flat_wt), len(flat_mut))
        flat_wt = np.pad(flat_wt, (0, max_len - len(flat_wt)))
        flat_mut = np.pad(flat_mut, (0, max_len - len(flat_mut)))
        
        results['SAVER3']['cosine'].append(cosine(flat_wt, flat_mut))
        results['SAVER3']['entropy'].append(entropy([q.norm() for q in qdna.additive_walk(quats_wt[:min_len])]))
        results['SAVER3']['time'].append(time.time() - start)
        
        # CGR
        start = time.time()
        cgr_wt = chaos_game_representation(wt)
        cgr_mut = chaos_game_representation(mut)
        results['CGR']['cosine'].append(cosine(cgr_wt.flatten(), cgr_mut.flatten()))
        results['CGR']['entropy'].append(entropy(cgr_wt.flatten() + 1e-10))
        results['CGR']['time'].append(time.time() - start)
        
        # Z-curve
        start = time.time()
        z_wt = z_curve(wt)
        z_mut = z_curve(mut)
        min_len = min(len(z_wt), len(z_mut))
        flat_wt = z_wt[:min_len].flatten()
        flat_mut = z_mut[:min_len].flatten()
        max_len = max(len(flat_wt), len(flat_mut))
        flat_wt = np.pad(flat_wt, (0, max_len - len(flat_wt)))
        flat_mut = np.pad(flat_mut, (0, max_len - len(flat_mut)))
        
        results['Z-curve']['cosine'].append(cosine(flat_wt, flat_mut))
        hist_wt = np.histogram(z_wt[:, 0], bins=20)[0] + 1e-10
        results['Z-curve']['entropy'].append(entropy(hist_wt / hist_wt.sum()))
        results['Z-curve']['time'].append(time.time() - start)
    
    return results

# ====================== VISUALIZATION ======================
def plot_results(results):
    # Convert results to pandas DataFrame for easier plotting
    metrics = ['cosine', 'entropy', 'time']
    data = []
    for method in results:
        for i in range(len(results[method]['cosine'])):
            for metric in metrics:
                data.append({
                    'Method': method,
                    'Metric': metric,
                    'Value': results[method][metric][i]
                })
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(18, 5))
    
    # Plot each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(
            data=df[df['Metric'] == metric],
            x='Method',
            y='Value',
            showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}
        )
        if metric == 'cosine':
            plt.title('Cosine Distance (Lower = Better)')
            plt.ylabel('Distance')
        elif metric == 'entropy':
            plt.title('Sequence Entropy')
            plt.ylabel('Entropy')
        else:
            plt.title('Runtime')
            plt.ylabel('Seconds per Sequence')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("Simulating 100 WT-mutant sequence pairs...")
    sequences = simulate_sequences(num_pairs=100)
    
    print("Running benchmark...")
    results = run_benchmark(sequences)
    
    print("Generating visualizations...")
    plot_results(results)
    
    # Print summary statistics
    print("\n=== Benchmark Summary ===")
    for method in results:
        print(f"\nMethod: {method}")
        print(f"  Mean Cosine Distance: {np.mean(results[method]['cosine']):.4f}")
        print(f"  Mean Entropy: {np.mean(results[method]['entropy']):.4f}")
        print(f"  Mean Runtime: {np.mean(results[method]['time']):.6f} sec/seq")



# ====================== STATISTICAL VALIDATION (Paired Comparisons) ======================
# This section assumes a 'results' dict exists as produced by run_benchmark(sequences),
# where results[method]['cosine'] is a list/array of cosine distances per pair for each method.
try:
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_rel, wilcoxon, shapiro, rankdata
    
    def paired_effect_size(diff):
        """Cohen's d for paired samples: mean(diff)/sd(diff)."""
        diff = np.asarray(diff, dtype=float)
        return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

    def wilcoxon_effect_size_r(statistic, n):
        """Approximate r = Z/sqrt(N) for Wilcoxon signed-rank (uses large-sample normal approx)."""
        # SciPy's wilcoxon returns the T statistic; we approximate Z via normal approximation.
        # For practical reporting, we transform p-value to Z using inverse normal CDF.
        from mpmath import erfcinv, sqrt
        # p = 2 * (1 - Phi(|Z|)) -> |Z| = sqrt(2) * erfcinv(p)
        # But we don't have p here, so caller should compute from p.
        return None  # Placeholder if Z not provided

    def bh_fdr(pvals, alpha=0.05):
        """Benjamini-Hochberg FDR correction. Returns rejected mask and adjusted p-values."""
        p = np.asarray(pvals, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        adj = np.empty(n, dtype=float)
        cummin = 1.0
        for i in range(n-1, -1, -1):
            cummin = min(cummin, ranked[i] * n / (i+1))
            adj[i] = cummin
        # Undo ordering
        adj_full = np.empty(n, dtype=float)
        adj_full[order] = adj
        reject = adj_full < alpha
        return reject, adj_full

    def perform_stat_validation(results, out_prefix="benchmark_stats"):
        methods = list(results.keys())
        metrics = {m: np.asarray(results[m]['cosine'], dtype=float) for m in methods}
        # Ensure equal lengths
        lens = [len(v) for v in metrics.values()]
        if len(set(lens)) != 1:
            raise ValueError(f"Cosine arrays have different lengths: {lens}")
        n = lens[0]

        pairs = [("SAVER3", "CGR"), ("SAVER3", "Z-curve"), ("CGR", "Z-curve")]
        rows = []
        pvals = []
        names = []
        
        for a, b in pairs:
            if a not in metrics or b not in metrics:
                continue
            diff = metrics[a] - metrics[b]
            # Normality check
            try:
                W, p_norm = shapiro(diff)
                normal = p_norm > 0.05
            except Exception:
                # For large N or constant diffs, Shapiro can fail; fall back to nonparametric
                normal = False
                p_norm = np.nan
            
            if normal:
                stat, p = ttest_rel(metrics[a], metrics[b])
                test_used = "paired t-test"
                eff = paired_effect_size(diff)
                eff_label = "Cohen_d_paired"
            else:
                stat, p = wilcoxon(metrics[a], metrics[b], zero_method="wilcox", alternative="two-sided", mode="auto")
                test_used = "Wilcoxon signed-rank"
                eff = np.nan  # r effect size can be derived from Z-score, but SciPy doesn't return Z directly.
                eff_label = "r_effect_size"
            
            rows.append({
                "comparison": f"{a} vs {b}",
                "n_pairs": n,
                "normality_p": p_norm,
                "test": test_used,
                "statistic": float(stat),
                "p_value": float(p),
                eff_label: float(eff) if eff==eff else np.nan,  # keep NaN if not computed
                "mean_diff": float(np.mean(diff)),
                "std_diff": float(np.std(diff, ddof=1)),
            })
            pvals.append(float(p))
            names.append(f"{a} vs {b}")

        # Multiple comparisons correction (BH-FDR)
        if pvals:
            reject, p_adj = bh_fdr(pvals, alpha=0.05)
            for i, row in enumerate(rows):
                row["p_adj_BH"] = float(p_adj[i])
                row["reject_at_0.05_FDR"] = bool(reject[i])

        df = pd.DataFrame(rows)
        import os
        csv_path = os.path.join(os.getcwd(), f"{out_prefix}_paired_stats.csv")
        df.to_csv(csv_path, index=False)
        
        print("\\n=== Paired Statistical Validation (Cosine Distance) ===")
        if df.empty:
            print("No valid method pairs found for statistical testing.")
        else:
            with pd.option_context("display.max_columns", None):
                print(df.to_string(index=False))
            print(f"\\nSaved detailed results to: {csv_path}")
        return df, csv_path

    # Auto-run if 'results' exists in the main namespace
    try:
        _ = results  # noqa: F821
        print("\\n[Statistical Validation] Detected 'results' dict — running paired tests...")
        _df_stats, _csv_stats = perform_stat_validation(results, out_prefix="benchmark_stats")
    except NameError:
        print("\\n[Statistical Validation] 'results' not found. Run the benchmark first to produce results, then call perform_stat_validation(results).")

except Exception as e:
    print("[Statistical Validation] Skipped due to error:", e)
