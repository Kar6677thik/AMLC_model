# Third-party components

This pipeline uses no external task dataset or pretrained neural checkpoint.

| Component | Purpose | License reference |
| --- | --- | --- |
| LightGBM | Gradient-boosted pair classifier | [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) |
| XGBoost (optional) | CUDA tree training and scoring | [Apache-2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE) |
| RapidFuzz | Local string similarity | [MIT](https://github.com/rapidfuzz/RapidFuzz/blob/main/LICENSE) |
| NumPy | Numeric arrays and disk-backed training matrices | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Python standard library / SQLite | TSV I/O, manifests, indexes, packaging | Installed distributions' notices |

The challenge's final-model license rule is distinct from dependency licenses. The trained baseline is distributed under the adjacent MIT license. Preserve the installed dependencies' license/notice files when redistributing their binaries. The final package records the actual compute-PC environment in `requirements.txt` and `assets/environment.txt`; this does not mean the development bootstrap ranges were previously tested.

The repository's original organizer statement, validator, template and challenge data are not relicensed by this project.
