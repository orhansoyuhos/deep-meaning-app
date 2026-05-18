# DeepMeaning App

A desktop GUI that estimates **local scene meaning maps** for
natural images using the DeepMeaning model (CoCa vision-language transformer
+ linear ensemble) from Hayes & Henderson (2025).

## Acknowledgements & citation

The underlying DeepMeaning model — the CoCa-based feature extraction, the
patch grid, the leave-one-scene-out linear ensembles for indoor and outdoor
scenes, and the released model weights — is the work of Hayes & Henderson
(2025). All of the modelling credit belongs to them; this repository only
adds a user interface on top of code they made publicly available on OSF
(https://osf.io/hcnfx).

**If you use this tool, please cite the following papers:**

> Hayes, T. R., & Henderson, J. M. (2025). DeepMeaning: Estimating and
> Interpreting Scene Meaning for Attention Using a Vision-Language
> Transformer. *Open Mind, 9*, 1020–1036.
> https://doi.org/10.1162/opmi.a.6

> Soyuhos, O., Hayes, T. R., Hu, W., Hamel, T. P., Sevak, B., Henderson,
> J. M., & Chen, X. (2026). Meaning-based guidance of attention in rhesus
> monkeys during naturalistic scene viewing. *bioRxiv*, 2026-03.
> https://doi.org/10.64898/2026.03.11.711223

## What the app does

Given a folder of same-sized scene images, the app:

1. Tiles each scene into 128×128 px patches at 73% overlap.
2. Runs every patch through a pretrained OpenCLIP CoCa ViT-L/14 to obtain
   image embeddings.
3. Predicts a per-patch meaning value using the indoor or outdoor
   ensemble linear model.
4. Reassembles the patch predictions into a full-resolution meaning map
   per scene, saved as a compressed `.npz`.
5. (GUI only) Pops up an interactive viewer that overlays the map on the
   original image with adjustable alpha and colormap.

The first run downloads the CoCa weights (a few GB) into the OpenCLIP
cache. A CUDA GPU is strongly recommended; CPU works but is much slower.

## Project layout

```
deep-meaning-app/
├── app/                          # the code root
│   ├── app.py                    # GUI entry point
│   ├── batch_deep_meaning.py     # CLI entry point
│   ├── environment.yml           # conda environment spec
│   ├── data/
│   │   ├── DeepMeaning/          # Hayes & Henderson ensemble weights (.pkl)
│   │   ├── grid/                 # patch grids for common image sizes (.npz)
│   │   └── patch_temp/           # scratch space for patches during a run
│   ├── deepmeaning/              # importable Python package
│   │   ├── paths.py              # central path / constant definitions
│   │   ├── coca.py               # OpenCLIP CoCa feature extraction
│   │   ├── utils.py              # original utility module
│   │   ├── pipeline.py           # run_meaning_map_job + dataclasses
│   │   └── gui/                  # Tkinter GUI (main window + viewer)
│   └── scripts/
│       ├── generate_grid.py      # build a new patch grid for an arbitrary size
│       └── visualize_map.py      # meaning map viewer
├── input/                        # GUI default input folder
└── output/                       # GUI default output folder
```

## Installation

This was developed on Windows with miniconda + an NVIDIA GPU
(CUDA 11.8). Other platforms should work but are not tested.

```bash
# from the app/ folder
conda env create -f environment.yml
conda activate meaning
```

The environment installs PyTorch with CUDA 11.8, OpenCLIP, OpenCV,
matplotlib, scikit-learn, scipy, pandas, Pillow, tqdm, and Tk.

## Usage

### GUI

```bash
conda activate meaning
cd app
python app.py
```

In the window:

1. Pick an **Input folder** (all images must be the same size).
2. Pick an **Output folder**.
3. Choose **Indoor** or **Outdoor** scene category.
4. Click **Generate Meaning Maps**.

If no grid exists for the image size, the app offers to generate one.
When the run finishes, the results viewer opens automatically — use the
arrow buttons to flip through images, the alpha slider to dial overlay
opacity, and the dropdown to switch between colormaps.

## Working with the output

Each saved `.npz` contains one 2-D `float64` array under the key `array`
matching the input image's `(H, W)`. Quick loading example:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("input/familiar_83.png")
m = np.load("output/familiar_83.npz")["array"]

plt.imshow(img)
plt.imshow(m, alpha=0.6, cmap="inferno")
plt.colorbar(label="meaning")
plt.show()
```

## Notes & limitations

- **Same-size input**. All images in a single run must share dimensions
  (the linear model is tied to a fixed patch grid). For mixed-size sets,
  pad images to a common size or process them in separate runs.
- **Indoor vs. outdoor**. Hayes & Henderson trained separate ensembles
  for indoor and outdoor scenes.
- **CoCa weights**. The first invocation downloads
  `coca_ViT-L-14` / `mscoco_finetuned_laion2b_s13b_b90k` via OpenCLIP.
  Allow disk space for the cache (default `~/.cache/clip`).

## License

The DeepMeaning model code distributed by Hayes & Henderson on OSF carries
their original license — please check the source repository
(https://osf.io/hcnfx) before redistribution.