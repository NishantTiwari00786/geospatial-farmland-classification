# Geospatial Farmland Classification — SAM Segmentation + CNN Pipeline

An end-to-end satellite imagery pipeline that combines Meta's Segment Anything Model (SAM) for automated region segmentation with a CNN classifier to distinguish farmland from metropolitan areas. Built during a research internship at the UCR Geospatial Lab under Dr. Ahmed Eldawy (MSRIP, Summer 2024).

**Author:** Nishant Tiwari

**Lab:** [UCR Geospatial Lab](https://www.cs.ucr.edu/~eldawy/) — Dr. Ahmed Eldawy

**Presentation:** [MSRIP Research Presentation](https://bit.ly/Nishant_MSRIP)

---

## Key Results

| Metric | Result |
|---|---|
| **Classification accuracy** | 85% on 100,000 satellite images |
| **Task** | Binary classification: farmland vs. metropolitan |
| **Raw dataset** | 2 TB+ Planet Labs satellite imagery (GeoTIFF), clipped and tiled for training efficiency |
| **Pipeline** | Tile extraction → SAM segmentation → polygon extraction → CNN classification |

---

## Pipeline

### Step 1 — Tile Extraction

Large-scale satellite GeoTIFFs are split into smaller, uniform tiles for processing. Raw Planet Labs imagery at full resolution is too large to process directly, so `Tille_extraction.py` breaks it into manageable sub-regions while preserving geospatial metadata.

### Step 2 — Segmentation with SAM

Meta's Segment Anything Model identifies and outlines distinct regions within each tile — field boundaries, road edges, building footprints, vegetation clusters. Two implementations:
- `SAM_on_OPENCV.py` — SAM + OpenCV for preprocessing, mask generation, and visualization
- `SAM_on_Tensorflow.py` — SAM + TensorFlow for integration with the downstream classification pipeline

### Step 3 — Polygon Extraction

`Extraction_polygon.py` converts SAM's segmentation masks into vector polygons from geospatial shapefiles, preparing structured region boundaries for classification.

### Step 4 — CNN Classification

A convolutional neural network trained on the segmented tiles classifies each region as farmland or metropolitan.
- `cnn_model.py` — model architecture, data augmentation, training pipeline
- `Testing_CNN.py` — evaluation with accuracy, precision, recall, and confusion matrix
- `Moving_dataset.py` — automated dataset organization into train/val splits with positive (farmland) and negative (metropolitan) labels

**Result:** 85% classification accuracy on 100,000 images.

---

## Repository Structure

```
geospatial-farmland-classification/
├── src/
│   ├── Tille_extraction.py         # Tile extraction from large GeoTIFFs
│   ├── SAM_on_OPENCV.py            # SAM segmentation with OpenCV
│   ├── SAM_on_Tensorflow.py        # SAM segmentation with TensorFlow
│   ├── Extraction_polygon.py       # Polygon extraction from segmentation masks
│   ├── cnn_model.py                # CNN architecture and training
│   ├── Testing_CNN.py              # Model evaluation and metrics
│   └── Moving_dataset.py           # Dataset preparation and splitting
├── ReadMe/                         # Additional documentation
└── README.md
```

## Tech Stack

Python · TensorFlow · OpenCV · Meta SAM · NumPy · Rasterio · GeoPandas · Shapely · Planet Labs Satellite Imagery (GeoTIFF)

## Acknowledgements

Research conducted at the UCR Geospatial Lab under the supervision of Dr. Ahmed Eldawy as part of the Multi-campus Summer Research Internship Program (MSRIP), Summer 2024.
