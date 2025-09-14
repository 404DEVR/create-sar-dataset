# Camping Site Suitability Dataset Generator

A Python script for generating machine learning datasets from multiple satellite images for camping site suitability prediction. This tool extracts overlapping patches and calculates various features including SAR-derived features using G0 distribution analysis.

## Features

- **Patch Extraction**: Extracts overlapping 50x50 pixel patches with 25-pixel stride (50% overlap)
- **SAR Feature Extraction**: Calculates alpha, gamma, and sigma values using G0 distribution analysis (Python equivalent of MATLAB code)
- **Terrain Features**: Extracts elevation and slope statistics
- **Texture Features**: Uses GLCM (Gray-Level Co-occurrence Matrix) for texture analysis
- **Spatial Features**: Calculates distance to water bodies
- **Quality Control**: Implements noise filtering and Lee filter for speckle reduction
- **Data Validation**: Comprehensive validation with outlier detection and statistics

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone or download the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from camping_site_dataset_generator import CampingSiteDatasetGenerator

# Initialize the generator
generator = CampingSiteDatasetGenerator(
    patch_size=50,           # 50x50 pixel patches
    stride=25,               # 25-pixel stride (50% overlap)
    suitability_threshold=0.6,  # 60% of pixels must be suitable
    noise_threshold=0.8,     # Maximum noise level threshold
    lee_filter_threshold=0.6 # Apply Lee filter for moderate noise
)

# Generate dataset
df, metadata = generator.generate_dataset(
    sar_path='sar_image.tif',
    dem_path='dem_image.tif',
    slope_path='slope_image.tif',
    landcover_path='landcover_image.tif',
    label_path='label_mask.tif',
    coherence_path='coherence_image.tif',  # Optional
    output_dir='output'
)
```

### Example with Sample Data

Run the example script to test with generated sample data:

```bash
python scripts/example_usage.py
```

## Input Images

The script requires the following satellite images:

1. **SAR Image**: Single band containing backscatter values
2. **DEM Image**: Elevation values in meters
3. **Slope Image**: Slope values in degrees
4. **Landcover Image**: Classified land use types
5. **Label Mask**: Binary image (1=suitable for camping, 0=not suitable)
6. **Coherence Image**: SAR coherence values 0-1 (optional)

## Output

The script generates:

1. **CSV Dataset**: `camping_site_dataset.csv` with the following columns:
   - `patch_id`: Unique identifier (format: "patch_row_col")
   - `alpha_mean`: Average alpha decomposition value
   - `gamma_mean`: Average gamma value
   - `sigma_mean`: Average sigma backscatter coefficient
   - `coherence_mean`: Average coherence value (median for robustness)
   - `dem_median`: Median elevation
   - `slope_median`: Median slope angle
   - `texture_contrast`: GLCM contrast measure
   - `texture_dissimilarity`: GLCM dissimilarity measure
   - `texture_homogeneity`: GLCM homogeneity measure
   - `distance_to_water`: Distance to nearest water body
   - `noise_level`: Coefficient of variation of SAR patch
   - `suitability_ratio`: Fraction of pixels labeled as suitable
   - `label`: Binary suitability label (1=suitable, 0=not suitable)

2. **Metadata File**: `dataset_metadata.json` containing:
   - Processing parameters
   - Image statistics
   - Dataset statistics
   - Feature descriptions

## Key Features

### SAR Feature Extraction

The script implements Python equivalents of your MATLAB G0 distribution analysis:

- **G0 Distribution Fitting**: Uses scipy.optimize.fsolve to solve the G0 parameter estimation
- **Alpha Calculation**: Extracted from G0 distribution parameters
- **Gamma Calculation**: Derived from alpha and mean intensity
- **Sigma Calculation**: Mean backscatter coefficient
- **Noise Level**: Coefficient of variation for quality control

### Quality Control

- **Noise Filtering**: Removes patches with noise_level > 0.8
- **Lee Filter**: Applied to the entire SAR image before patch extraction if noise level >= 0.6
- **Data Validation**: Checks for NaN values, outliers, and data consistency

### Memory Efficiency

- **Progress Tracking**: Shows extraction progress with tqdm
- **Intermediate Saves**: Saves results every 1000 patches
- **Batch Processing**: Processes patches in batches to manage memory

## Configuration

You can customize the following parameters:

```python
generator = CampingSiteDatasetGenerator(
    patch_size=50,              # Patch size (default: 50x50)
    stride=25,                  # Stride for overlap (default: 25)
    suitability_threshold=0.6,  # Suitability ratio threshold (default: 0.6)
    noise_threshold=0.8,        # Maximum noise level (default: 0.8)
    lee_filter_threshold=0.6    # Lee filter threshold (default: 0.6)
)
```

## Integration with MATLAB Code

The script provides Python equivalents of your MATLAB functions:

- `g0fun.m` → `g0_fun()` method
- `g0molc.m` → `g0molc()` method
- G0 distribution parameter estimation using scipy.optimize.fsolve

## Error Handling

The script includes comprehensive error handling for:
- Missing or corrupted image files
- Dimension mismatches between images
- Optimization failures in G0 parameter estimation
- Memory issues during processing
- Invalid patch data

## Performance

- **Progress Tracking**: Real-time progress updates
- **Memory Management**: Efficient processing for large images
- **Intermediate Saves**: Prevents data loss during long processing
- **Validation**: Comprehensive data quality checks

## Example Output

```
Starting camping site dataset generation...
Loading satellite images...
Images loaded successfully. Dimensions: (500, 500)
Applying Lee filter to SAR image...
SAR image noise level: 0.723 (>= 0.6)
Applying Lee filter to reduce speckle noise...
Lee filter applied successfully
Extracting patches...
Total patches to extract: 361
Extracting patches: 100%|████████████| 361/361 [00:45<00:00,  8.02it/s]
Extracted 298 valid patches
Validating data...
Validation complete:
  Total patches: 298
  NaN values: 0
  Label distribution: {0: 201, 1: 97}
  Outliers detected: 12

Dataset generation complete!
Dataset saved to: camping_site_output/camping_site_dataset.csv
Metadata saved to: camping_site_output/dataset_metadata.json
Total patches: 298
Class distribution: {0: 201, 1: 97}
```

## License

This project is based on MATLAB G0-MAP code and provides Python equivalents for satellite image analysis and machine learning dataset generation.

## Contributing

Feel free to submit issues and enhancement requests!