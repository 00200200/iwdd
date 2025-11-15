# [IWDDCONTEST](https://mivia.unisa.it/iwddcontest2026/)

## Requirements

- **Python 3.11+**
- **uv package manager** - https://docs.astral.sh/uv/getting-started/installation/

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/00200200/iwdd.git
cd iwdd

# Install dependencies
uv sync
```

### 2. Prepare Dataset

Download the dataset from Google Drive and using a script.

uv run python -m src.api_integration.file_servers.download_data_gdrive

### 3. Training

**Basic training with default parameters:**

```bash
# Train default model (VideoMAE-SSv2)
uv run -m src.scripts.train

# Train specific model
uv run -m src.scripts.train --model videmoae_ssv2_short --lr 1e5 --epochs 10 --batch-size 8 --unfreeze-layers 1 --clip-duration 3 --stride 1 --accelerator cuda
uv run -m src.scripts.train --model videomae_ssv2
uv run -m src.scripts.train --model xclip
uv run -m src.scripts.train --model videomae_kinetics
```

**Available models:** `videomae_ssv2` `videomae_ssv2_shrot`, `videomae_kinetics`, `xclip` (configured in `config/models_config.yaml`)

**CLI arguments:**

```bash
--model              # Model name (default: videomae_ssv2)
--epochs             # Number of epochs (default: 10)
--batch-size         # Batch size (default: 8)
--lr                 # Learning rate (default: 1e-5)
--unfreeze-layers    # Layers to unfreeze (default: 1, use 0 for classifier only)
--clip-duration      # Clip duration in seconds (default: 3)
--stride             # Sliding window stride (default: 1)
--accelerator        # Device: auto/cuda/mps/cpu (default: auto)
```

**Code structure** (`src/scripts/train.py`):

```python
# Model loaded from config
model_config = get_model_config(args.model)

model = VideoClassificationModel(
    model_config=model_config,        # Config from YAML
    learning_rate=1e-5,               # Learning rate
    num_unfreeze_layers=1,            # Number of layers to unfreeze
)

# Data parameters
data = IWDDDataModule(
    model_config=model_config,        # Same config
    videos_dir="data/raw/videos",     # Path to videos
    annotations_dir="data/raw/labels", # Path to annotations
    batch_size=8,                      # Batch size
    num_workers=4,                     # Number of data loading workers
    clip_duration=3,                   # Clip duration in seconds
    stride=1,                          # Stride for sliding window
    num_frames=16,                     # Number of frames per clip
    train_split=0.7,                   # Training set
    val_split=0.15,                    # Validation set
)

# Trainer parameters
trainer = L.Trainer(
    max_epochs=10,           # Number of epochs
    accelerator="auto",      # Accelerator
    log_every_n_steps=1,     # Logging frequency
)
```

### 4. Monitor Training

During training, metrics are logged to TensorBoard:

```bash
tensorboard --logdir lightning_logs/
```

Open http://localhost:6006 in your browser to view:

### 5. Checkpoints

Best models are saved to `lightning_logs/version_X/checkpoints/` based on validation loss.

## Project Structure

```
iwdd/
├── src/
│ ├── data/
│ │ └── dataset.py # Dataset and DataModule
│ ├── model/
│ │ └── model.py # model definition
│ ├── scripts/
│ │ └── train.py # Training script
│ └── utils/
│ ├── metrics.py # Evaluation metrics
│ └── utils.py # Utility functions
├── data/
│ └── raw/ # Dataset directory
├── docs/ # Documentation
│ └── YOLO.md # YOLO training guide
├── lightning_logs/ # Training logs and checkpoints
└── README.md
```

## Documentation

- **[YOLO Training Guide](docs/YOLO.md)** - Instructions for training YOLO models with multiple datasets
