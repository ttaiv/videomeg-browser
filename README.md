# Video browser extension for MNE Python's qt raw data browser

## Installation for development

This project does not have any dependencies outside of MNE Python. However, to
avoid interferene with your stable MNE Python installation, it is recommended to
install in a separate environment. Here's how to achieve this with `conda`:

1. Create a new conda environment (named `mne-videomeg`) with MNE Python installed:

   ```bash
   conda create --channel=conda-forge --strict-channel-priority --name=mne-videomeg mne
   ```

2. Clone this repository and navigate to project root.

3. Activate the environment:

   ```bash
   conda activate mne-videomeg
   ```

4. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

Now you should be able to test the video browser by running example scripts in `scripts/` directory. For example:

```bash
python scripts/run_sync_demo.py
```

Note that you need to adjust the paths in the example scripts to point to your video files and raw data files.
