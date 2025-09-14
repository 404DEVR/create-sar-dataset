# Project Structure

This document describes the organization of the SAR Camping Site Dataset Generator project.

## 📁 Folder Structure

```
create-sar-dataset/
├── 📁 data/                          # Satellite image data
│   ├── SAR_Image_5k_Tso_Moriri.tif
│   ├── DEM_5km_Tso_Moriri.tif
│   ├── Slope_5k_Tso_Moriri.tif
│   ├── Landcover_Image_5k_Tso_Moriri.tif
│   ├── labels_Tso_Moriri.png
│   └── Coherence_Stability_5km__Tso_Moriri.tif
├── 📁 scripts/                       # Utility scripts
│   ├── example_usage.py
│   └── test_water_detection.py
├── 📁 camping_site_output/           # Generated datasets
│   ├── camping_site_dataset.csv
│   └── dataset_metadata.json
├── 📁 output/                        # General output folder
├── 📁 results/                       # Analysis results
├── 📁 models/                        # Trained ML models
├── 📁 venv/                          # Virtual environment
├── 🐍 camping_site_dataset_generator.py  # Main script
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Project documentation
├── 📄 PATCH_FILTERING_EXPLANATION.md # Filtering details
├── 📄 PROJECT_STRUCTURE.md           # This file
└── 📄 .gitignore                     # Git ignore rules
```

## 📂 Folder Descriptions

### `data/`
Contains all satellite imagery and input data files:
- **SAR images**: Synthetic Aperture Radar data (.tif)
- **DEM files**: Digital Elevation Model data (.tif)
- **Slope data**: Terrain slope information (.tif)
- **Landcover**: Land use classification data (.tif)
- **Labels**: Ground truth camping site suitability (.png)
- **Coherence**: SAR coherence data (.tif)

### `scripts/`
Utility and example scripts:
- **example_usage.py**: Demonstrates how to use the generator
- **test_water_detection.py**: Tests water body detection algorithms

### `camping_site_output/`
Default output location for generated datasets:
- **camping_site_dataset.csv**: Main ML dataset
- **dataset_metadata.json**: Processing metadata and statistics

### `output/`
General purpose output folder for custom results

### `results/`
Analysis results, plots, and evaluation metrics

### `models/`
Trained machine learning models and checkpoints

## 🚀 Usage with New Structure

### Basic Usage
```bash
# Run with default data paths
python camping_site_dataset_generator.py

# Run with custom paths
python camping_site_dataset_generator.py \
  --sar data/SAR_Image_5k_Tso_Moriri.tif \
  --dem data/DEM_5km_Tso_Moriri.tif \
  --slope data/Slope_5k_Tso_Moriri.tif \
  --landcover data/Landcover_Image_5k_Tso_Moriri.tif \
  --label data/labels_Tso_Moriri.png \
  --coherence data/Coherence_Stability_5km__Tso_Moriri.tif \
  --output results/my_dataset
```

### Testing Scripts
```bash
# Test water detection
python scripts/test_water_detection.py

# Run example
python scripts/example_usage.py
```

## 📋 Benefits of This Structure

1. **Clean Root Directory**: Main scripts and documentation at top level
2. **Organized Data**: All satellite images in dedicated `data/` folder
3. **Separated Outputs**: Different output types in appropriate folders
4. **Utility Scripts**: Helper scripts organized in `scripts/` folder
5. **Version Control**: `.gitignore` properly configured for each folder type
6. **Scalability**: Easy to add new data sources or output types

## 🔧 Configuration

The main script automatically uses the new folder structure:
- Default data paths point to `data/` folder
- Default output goes to `camping_site_output/`
- All paths are configurable via command line arguments

## 📝 Adding New Data

To add new satellite images:
1. Place files in the `data/` folder
2. Update file paths in script or use command line arguments
3. Ensure consistent naming convention for easy identification