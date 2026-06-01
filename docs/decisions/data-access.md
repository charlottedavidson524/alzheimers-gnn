Original plan:

- DataLad clone and `datalad get` for selective downloads. This was recommended in the PEARL-Neuro paper and on the OpenNeuro dataset page.

What changed:

- Switched to direct HTTPS downloads using a Python script (`scripts/download_data.py`) targeting OpenNeuro’s public S3 URLs.

Why:

- DataLad on Windows needs git-annex, which is not available via conda-forge for Windows. The official Windows installer triggered SmartScreen warnings that couldn’t be bypassed on the local machine.
- Provenance is preserved in other ways. Dataset version is recorded in `config/default.yaml`, and the exact file list downloaded is version-controlled in `scripts/download_data.py`. This is enough for reproducibility and methods-chapter documentation.

Alternatives considered:

- AWS CLI (aws s3 cp/sync). Reliable and standard. Rejected because it adds a tool to install and learn but has no advantage over direct HTTPS (only uses the Python standard library).
- Azure ML compute/storage: university access available, but unsure about limits of cloud compute and storing data. Downloading via the cloud isn’t necessary for a local-first development workflow.
- University HPC cluster. Appropriate later for parallel EEG preprocessing, hyperparameter tuning and fMRI work. Not suited for downloading (network-bound, not compute-bound) or for early development (cluster friction outweighs benefit at this stage).
- OpenNeuro Node.js CLI. Official tool but requires installing Node.js. This would be an additional runtime. Rejected for the same reason as AWS CLI (no advantage over plain Python).
- OpenNeuro web interface. Used for one-off downloads of small metadata files (e.g. participants.tsv). Not suitable for the full dataset.

Implications:

- No structural changes to the project. `config/`, `src/`, and `pyproject.toml` don’t need to change. Data still lives outside the repo at `C:/data/pearl-neuro/` (matches config/default.yaml).
- `scripts/download_data.md` updated to document HTTPS method.
- `scripts/download_data.py` added. Is the runnable downloader. Resumable (skips existing files), handles 404s gracefully (some files are missing in the public release, see PEARL-Neuro paper), and structured in stages so downloads can be expanded incrementally (metadata -> one test subject (testing download works) -> sample subjects (testing methodology) -> full dataset).
- DataLad and git-annex can be left installed (harmless). Aren’t relied on by any of the project code.
