# iwdd

## Requirements

- **Python 3.11+**
- **uv package manager** - https://docs.astral.sh/uv/getting-started/installation/

## Quick Start

1. **Setup:**

   ```bash
   # Clone the repository
   git clone https://github.com/00200200/iwdd.git
   cd iwdd

   # Install dependencies
   uv sync
   ```

# Preparing YOLO datasets

Make sure all datasets are located in ```datasets``` folder.
Currently we are using these datasets:
   1. [taco-trash-dataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)
   2. [Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images?resource=download)
   3. [WasteInsight: Dataset for Detection and Volume Estimation of Municipal Solid Waste](https://data.mendeley.com/datasets/p8n7nbxyw3/1)

Given varying dataset formatting and initial COCO format first we need to convert those sets into YOLO format:
### 1. taco-trash-dataset

From the root directory run:

```bash
uv run src/scripts/preprocess_taco.py

uv run coco_to_yolo datasets/_taco-trash-dataset_processed datasets_yolo/ --test_ratio 0.15 --val_ratio 0.1

mv datasets_yolo/converted datasets_yolo/taco-trash-dataset_yolo && mv datasets_yolo/converted.yaml datasets_yolo/taco-trash-dataset_yolo/classes.yaml 
```

### 2. Synthetic Bags

```bash
uv run src/scripts/preprocess_synthetic_bags.py
uv run coco_to_yolo datasets/_synthetic-bags_processed datasets_yolo/ --test_ratio 0.15 --val_ratio 0.1
mv datasets_yolo/converted datasets_yolo/synthetic-bags_yolo && mv datasets_yolo/converted.yaml datasets_yolo/synthetic-bags_yolo/classes.yaml
```

### 3. WasteInsight

From the root directory run:
```bash
uv run src/scripts/preprocess_wasteinsight.py
uv run coco_to_yolo datasets/_wasteinsight_processed datasets_yolo/ --test_ratio 0.15 --val_ratio 0.1
mv datasets_yolo/converted datasets_yolo/wasteinsight_yolo && mv datasets_yolo/converted.yaml datasets_yolo/wasteinsight_yolo/classes.yaml
```
