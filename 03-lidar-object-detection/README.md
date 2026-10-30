# Sparse LiDAR 3D Object Detection — Four Architectures (A–D)

Research implementations for the paper **Cost-Sensitive Hybrid Transformer–Graph Neural Networks for Sparse LiDAR Object Detection in Automated Driving**, published in *International Journal of Intelligent Transportation Systems Research* (Springer).  
**DOI:** https://doi.org/10.1007/s13177-026-00677-y

## Architectures

| Variant | Architecture | Training script | Evaluation script |
|---|---|---|---|
| A | Transformer → FCN | `train_model_a.py` | `evaluate_model_a.py` |
| B | GNN → FCN | `train_model_b.py` | `evaluate_model_b.py` |
| C | GNN → Transformer → FCN | `train_model_c.py` | `evaluate_model_c.py` |
| D | Transformer → GNN → FCN | `train_model_d.py` | `evaluate_model_d.py` |

`extended_evaluation_model_d.py` contains additional complexity and detection-metric analyses for Model D.

The implementations include object classification and 3D bounding-box regression. They were developed for six classes (car, cyclist, human, wall, tree, cart). **Model D** is the principal architecture reported in the publication.

## Dependencies

See `requirements.txt`. Some optional visualization and graph libraries have fallbacks in individual scripts.

## Data availability and execution

**No dataset, point clouds, annotations, checkpoints, or internal experiment results are distributed here.** Access to the underlying research data is subject to BRIN authorization and the publication's data-availability statement. The scripts expect locally supplied, authorized inputs.

Original institution-specific absolute paths have been replaced with relative locations:

- `data/train`, `data/val`, `data/test`, `data/pcd_test`
- `outputs/checkpoints`, `outputs/results`, `outputs/figures`

These are **archival research scripts**, not a turnkey reproducibility package. They have passed Python syntax checks, but have **not** been retrained, rerun, or verified against the private dataset after path replacement. Some original scripts have experimental assumptions and checkpoint names that may require local configuration. No claims of independent reproduction are made.

## Permission and attribution

This repository is intended for research documentation and portfolio review. Publication of the paper does not automatically grant permission to publish source code. **Confirm permission with the BRIN supervisor or other rights holder before making the repository public.** Do not commit datasets, `.pcd` files, `.json` annotations, model weights, internal paths, or credentials.
