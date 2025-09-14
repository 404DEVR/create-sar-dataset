#!/usr/bin/env python3
"""
Example usage script for the Camping Site Dataset Generator.

This script demonstrates how to use the CampingSiteDatasetGenerator class
to generate a machine learning dataset from satellite images.
"""

import os
import numpy as np
from camping_site_dataset_generator import CampingSiteDatasetGenerator

def create_sample_data(output_dir='sample_data'):
    """
    Create sample satellite images for testing purposes.
    
    Args:
        output_dir (str): Directory to save sample images
    """
    print("Creating sample satellite images...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Image dimensions
    height, width = 500, 500
    
    # Generate sample SAR image (with speckle noise)
    np.random.seed(42)
    texture = np.random.exponential(1.0, (height, width))
    speckle = np.random.rayleigh(1.0, (height, width))
    sar_image = (texture * speckle * 1000).astype(np.uint16)
    
    # Generate sample DEM image (elevation in meters)
    dem_image = np.random.normal(1000, 200, (height, width)).astype(np.uint16)
    dem_image = np.clip(dem_image, 0, 3000)
    
    # Generate sample slope image (slope in degrees)
    slope_image = np.random.exponential(10, (height, width)).astype(np.uint16)
    slope_image = np.clip(slope_image, 0, 90)
    
    # Generate sample landcover image (1=water, 2=forest, 3=grassland, 4=urban)
    landcover_image = np.random.choice([1, 2, 3, 4], (height, width), p=[0.1, 0.3, 0.4, 0.2])
    
    # Generate sample label mask (1=suitable for camping, 0=not suitable)
    # Make camping suitability correlated with low slope and specific landcover
    suitable_mask = (slope_image < 15) & (landcover_image == 3)  # Low slope + grassland
    label_mask = suitable_mask.astype(np.uint8)
    
    # Generate sample coherence image
    coherence_image = np.random.beta(2, 2, (height, width)).astype(np.float32)
    
    # Save sample images
    import cv2
    
    cv2.imwrite(os.path.join(output_dir, 'SAR_Image_5k_Tso_Moriri.tif'), sar_image)
    cv2.imwrite(os.path.join(output_dir, 'DEM_5km_Tso_Moriri.tif'), dem_image)
    cv2.imwrite(os.path.join(output_dir, 'Slope_5k_Tso_Moriri.tif'), slope_image)
    cv2.imwrite(os.path.join(output_dir, 'Landcover_Image_5k_Tso_Moriri.tif'), landcover_image)
    cv2.imwrite(os.path.join(output_dir, 'label_mask_Tso_Moriri.tif'), label_mask)
    cv2.imwrite(os.path.join(output_dir, 'Coherence_Stability_5km__Tso_Moriri.tif'), 
                (coherence_image * 255).astype(np.uint8))
    
    print(f"Sample images created in {output_dir}/")
    return output_dir

def main():
    """
    Main function demonstrating dataset generation with Tso Moriri images.
    """
    print("Camping Site Dataset Generator - Tso Moriri Dataset")
    print("=" * 50)
    
    # Initialize the dataset generator with improved parameters
    generator = CampingSiteDatasetGenerator(
        patch_size=50,           # 50x50 pixel patches
        stride=25,               # 25-pixel stride (50% overlap)
        suitability_threshold=0.3,  # 30% of pixels must be suitable (reduced from 0.6)
        noise_threshold=1.2,     # Maximum noise level threshold (increased from 0.8)
        lee_filter_threshold=0.8 # Apply Lee filter for moderate noise (increased from 0.6)
    )
    
    # Define image paths - using your actual Tso Moriri images
    image_paths = {
        'sar_path': 'SAR_Image_5k_Tso_Moriri.tif',
        'dem_path': 'DEM_5km_Tso_Moriri.tif',
        'slope_path': 'Slope_5k_Tso_Moriri.tif',
        'landcover_path': 'Landcover_Image_5k_Tso_Moriri.tif',
        'label_path': 'label_mask_Tso_Moriri.tif',
        'coherence_path': 'Coherence_Stability_5km__Tso_Moriri.tif'
    }
    
    # Check if all image files exist
    missing_files = [path for path in image_paths.values() if not os.path.exists(path)]
    if missing_files:
        print(f"Missing files: {missing_files}")
        return
    
    try:
        # Generate the dataset
        print("\nGenerating dataset...")
        df, metadata = generator.generate_dataset(
            sar_path=image_paths['sar_path'],
            dem_path=image_paths['dem_path'],
            slope_path=image_paths['slope_path'],
            landcover_path=image_paths['landcover_path'],
            label_path=image_paths['label_path'],
            coherence_path=image_paths['coherence_path'],
            output_dir='camping_site_output'
        )
        
        # Display dataset summary
        print("\nDataset Summary:")
        print(f"Total patches: {len(df)}")
        print(f"Feature columns: {list(df.columns)}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        print(f"\nFeature statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['label', 'row', 'col']:
                print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        
        print(f"\nDataset saved to: camping_site_output/camping_site_dataset.csv")
        print(f"Metadata saved to: camping_site_output/dataset_metadata.json")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()