# iwdd

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

Download the dataset from Google Drive and organize it as follows:

```
iwdd/
└── data/
    └── raw/
        ├── labels/    # annotation files
        └── videos/    # video files
```

### 3. Training

**Basic training with default parameters:**

```bash
uv run -m src.scripts.train
```

**Available training parameters** (modify in `src/scripts/train.py`):

```python
model = VideoMAEModel(
    learning_rate=1e-4  # Learning rate
)

# Data parameters
data = IWDDDataModule(
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
    accelerator="cuda",      # acceleator
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
