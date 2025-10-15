# Preprocess Demo Assets

Use this folder to stage local inputs before running `scripts/preprocess.sh`.

1. Drop a driving video at `inputs/driving.mp4` (or edit the YAML specs to point elsewhere).
2. Provide a reference portrait at `inputs/reference.png`.
3. Run the preprocessing script from the repo root:
   ```bash
   ./scripts/preprocess.sh examples/preprocess_demo/animate_job.yaml --dump-resolved
   ```
   The processed assets will appear under `data/jobs/<job_id>/`.

> Media files are intentionally excluded from version control—add your own samples locally.
