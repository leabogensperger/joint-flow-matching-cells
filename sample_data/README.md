# sample_data/

Empty by default. This is where a small curated subset of **real** single-cell
images goes, so this repo can be tried out (training, VAE reconstruction,
flow-model morphing) without access to the full private dataset.

To populate it:

1. Copy a handful of real image files here, preserving their
   `<setting>/<filename>` folder structure from the real dataset, e.g.:
   ```
   sample_data/ONS76_DMSO_0uM/E07_T000_Z000_Cell0001.png
   sample_data/ONS76_C7_0.037uM/E07_T000_Z000_Cell0284.png
   ```
   `setting` is the same string as the `setting` column in
   `metadata/cell_lines_drug_compounds.csv`.
2. From the repo root, run:
   ```bash
   python scripts/build_sample_metadata.py
   ```
   This looks each file up in `metadata/cell_lines_drug_compounds.csv` and
   writes `metadata/sample_metadata.csv` with the matching real labels/splits.
3. Use `config/vae_microscopy_sample.yaml` / `config/latent_flow_joint_sample.yaml`
   to train on it, or pass `--data metadata/sample_metadata.csv --data_root
   sample_data` to `evaluate_vae.py` / `evaluate_latent_flow_joint.py` to
   evaluate any checkpoint (including a downloaded pretrained one) against
   it.

See the README's "Data" section for a suggested size.
