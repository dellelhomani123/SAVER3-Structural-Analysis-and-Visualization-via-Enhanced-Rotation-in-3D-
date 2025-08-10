[README.md](https://github.com/user-attachments/files/21704390/README.md)
# SAVER3 — Structural Analysis and Visualization via Enhanced Rotation in 3D

**Short description (≤350 chars):**  
Quaternion-based, alignment-free DNA analysis tool. SAVER3 encodes sequences as quaternion walks with sphere projections, using cosine distance and Shannon entropy to proportionally detect and visualize structural changes between wild-type and mutant genomes.

---

## Overview
**SAVER3** (Structural Analysis and Visualization via Enhanced Rotation in 3D) is a **quaternion-based framework** for DNA sequence representation, comparative genomics, and mutation analysis.

It encodes nucleotides as **unit quaternions**, generates cumulative **quaternion walks**, and projects them onto the **unit quaternion sphere**. This captures both magnitude and orientation changes in sequence structure.

The framework computes:
- **Cosine distance** — global directional divergence between sequences.
- **Shannon entropy** — nucleotide composition diversity.

These metrics, combined with quaternion projections, provide **stable, proportional, and interpretable** results across subtle and disruptive genomic variants.

---

## Features
- Alignment-free sequence analysis
- Quaternion-based encoding of DNA
- Norm trajectory visualization
- Unit-sphere projection
- Cosine similarity and Shannon entropy computation
- Benchmarking vs. Chaos Game Representation (CGR) and Z-curve
- Publication-ready figures and statistics

---

## Repository contents
```
SAVER3_Benchmark_TO_BE_PUBLISHED_2025_with_stats.py    # Benchmark + statistical validation
SAVER3_Benchmark_TO_BE_PUBLISHED_2025.py               # Benchmark without stats
SAVER3_DNA_ANALYSIS_METRICS_TO_BE_PUBLISHED_2025.py     # Entropy & cosine metrics for DNA sequences
SAVER3_GENOMIC_COMPARISON_TO_BE_PUBLISHED_2025.py      # Per-gene WT vs. mutant comparison
SAVER3_TO_BE_PUBLISHED_2025.py                         # Core SAVER3 quaternion DNA encoding
docs/SAVER3_Paper_2025_Final_9.docx                     # Related manuscript (optional)
requirements.txt                                        # Python dependencies
.gitignore                                              # Ignore rules for Python projects
```

---

## Installation
```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

## Usage Examples

### 1. Run core SAVER3 analysis
```bash
python SAVER3_TO_BE_PUBLISHED_2025.py
```
This processes DNA sequences, generates quaternion walks, and saves visualizations.

### 2. Compare WT vs. mutant sequences for selected genes
```bash
python SAVER3_GENOMIC_COMPARISON_TO_BE_PUBLISHED_2025.py
```

### 3. Compute DNA metrics (entropy & cosine distance)
```bash
python SAVER3_DNA_ANALYSIS_METRICS_TO_BE_PUBLISHED_2025.py
```

### 4. Run benchmarks against CGR and Z-curve
```bash
python SAVER3_Benchmark_TO_BE_PUBLISHED_2025_with_stats.py
```

---

## Requirements
- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Biopython
- scikit-learn
- umap-learn
- pandas

All dependencies are listed in `requirements.txt`.

---

## Citation
If you use this code in your research, please cite:

> Elhomani A. **SAVER3: Structural Analysis and Visualization via Enhanced Rotation in 3D**. University of Missouri–Kansas City, 2025.

BibTeX:
```bibtex
@misc{Elhomani2025SAVER3,
  author = {Abdellatif Elhomani},
  title = {SAVER3: Structural Analysis and Visualization via Enhanced Rotation in 3D},
  year = {2025},
  institution = {University of Missouri–Kansas City},
  url = {https://github.com/dellelhomani123/SAVER3-Structural-Analysis-and-Visualization-via-Enhanced-Rotation-in-3D-}
}
```

---

## License
© 2025 Abdellatif Elhomani. All rights reserved.
