# Batch Correction Metrics Report

## Model Information

- **z_i_dim**: 30
- **z_s_dim**: 5
- **n_cells**: 9025
- **n_genes**: 2443
- **n_batches**: 4
- **n_celltypes**: 23

## Metrics Explanation

**Batch ASW (Average Silhouette Width)**:
- Measures batch separation in the latent space
- Range: [-1, 1]
- **Lower is better** (closer to 0 means less batch effect)
- Negative values indicate good batch mixing

**Cell Type ASW**:
- Measures biological (cell type) separation
- Range: [-1, 1]
- **Higher is better** (closer to 1 means well-separated cell types)

**Batch Mixing Entropy**:
- Measures diversity of batches in k-NN neighborhoods
- Range: [0, log(n_batches)]
- **Higher is better** (more batch mixing)

---

## Results

### TRAIN Set

| Metric | Raw Data | Invariant (z_i) | Change | Interpretation |
|--------|----------|-----------------|--------|----------------|
| Batch ASW (↓ better) | 0.2361 | -0.0466 | -0.2827 | ✅ Improved |
| Cell Type ASW (preserve) | -0.3099 | 0.2737 | +0.5836 | ⚠️ Changed |
| Batch Mixing Entropy (↑ better) | 0.0183 | 0.9619 | +0.9436 | ✅ Improved |

### VAL Set

| Metric | Raw Data | Invariant (z_i) | Change | Interpretation |
|--------|----------|-----------------|--------|----------------|
| Batch ASW (↓ better) | 0.2427 | -0.0514 | -0.2942 | ✅ Improved |
| Cell Type ASW (preserve) | -0.3229 | 0.2319 | +0.5548 | ⚠️ Changed |
| Batch Mixing Entropy (↑ better) | 0.0576 | 1.0592 | +1.0016 | ✅ Improved |

---

## Summary

**Good batch correction should show**:
- ✅ Batch ASW decreases (closer to 0 or negative)
- ✅ Cell Type ASW remains high or increases
- ✅ Batch Mixing Entropy increases
- ✅ Similar metrics for train and val (good generalization)

