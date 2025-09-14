#!/usr/bin/env python3
"""
Test script to understand water detection in landcover data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio
from camping_site_dataset_generator import CampingSiteDatasetGenerator

def test_water_detection():
    # Load landcover image
    with rasterio.open('../data/Landcover_Image_5k_Tso_Moriri.tif') as src:
        landcover = src.read(1)

    print('Landcover unique values:', np.unique(landcover))
    
    # Get overall statistics
    unique_vals, counts = np.unique(landcover, return_counts=True)
    print('\nOverall landcover distribution:')
    for val, count in zip(unique_vals, counts):
        print(f'  Value {val}: {count} pixels ({count/landcover.size*100:.2f}%)')

    # Test water detection on a few patches
    generator = CampingSiteDatasetGenerator()
    patch_size = 50

    # Test patches from different areas
    test_centers = [(100, 100), (200, 200), (300, 300), (400, 400)]

    for i, (row, col) in enumerate(test_centers):
        print(f'\n--- Test Patch {i+1} at center ({row}, {col}) ---')
        patch = landcover[row:row+patch_size, col:col+patch_size]
        unique_vals, counts = np.unique(patch, return_counts=True)
        print(f'Patch unique values: {unique_vals}')
        print(f'Patch value counts: {counts}')
        
        # Test water detection
        spatial_features = generator.calculate_spatial_features(patch, (row, col))
        print(f'Distance to water: {spatial_features["distance_to_water"]:.2f}')
        print(f'Water code detected: {spatial_features.get("water_code_detected", "None")}')

if __name__ == "__main__":
    test_water_detection()
