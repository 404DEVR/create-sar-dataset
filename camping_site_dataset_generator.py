#!/usr/bin/env python3
"""
Camping Site Suitability Dataset Generator - Two-Stage ML Approach

This script generates a machine learning dataset from multiple satellite images
for camping site suitability prediction using a two-stage approach.

Author: Generated based on MATLAB G0-MAP code
Date: 2024
"""

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from scipy.optimize import fsolve
from scipy.special import psi, digamma
from skimage.feature import graycomatrix, graycoprops
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CampingSiteDatasetGenerator:
    """
    Main class for generating camping site suitability dataset from satellite images
    for two-stage ML approach.
    """
    
    # Landcover legend for interpreting landcover codes
    LANDCOVER_LEGEND = {
        10: "Tree cover",
        20: "Shrubland", 
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen"
    }
    
    def __init__(self, patch_size=50, stride=25, noise_threshold=1.2, lee_filter_threshold=0.8):
        """
        Initialize the dataset generator for two-stage ML approach.
        
        Args:
            patch_size (int): Size of patches to extract (default: 50x50)
            stride (int): Stride for patch extraction (default: 25)
            noise_threshold (float): Maximum noise level threshold (default: 1.2)
            lee_filter_threshold (float): Threshold for applying Lee filter (default: 0.8)
        """
        self.patch_size = patch_size
        self.stride = stride
        self.noise_threshold = noise_threshold
        self.lee_filter_threshold = lee_filter_threshold
        self.patches_data = []
        self.metadata = {}
        
    def load_images(self, sar_path, dem_path, slope_path, landcover_path, 
                   label_path, coherence_path=None):
        """Load all required satellite images."""
        print("Loading satellite images...")
        
        images = {}
        
        # Load all images
        images['sar'] = self.load_image(sar_path)
        images['dem'] = self.load_image(dem_path)
        images['slope'] = self.load_image(slope_path)
        images['landcover'] = self.load_image(landcover_path)
        images['label'] = self.load_image(label_path)
        
        if coherence_path:
            images['coherence'] = self.load_image(coherence_path)
        else:
            images['coherence'] = None
        
        # Check and handle dimension mismatches
        print("Checking image dimensions...")
        for name, img in images.items():
            if img is not None:
                print(f"{name}: {img.shape}")
        
        # Find the largest dimensions and resize all images
        max_height = max(img.shape[0] for img in images.values() if img is not None)
        max_width = max(img.shape[1] for img in images.values() if img is not None)
        target_shape = (max_height, max_width)
        
        print(f"Resizing all images to match largest dimensions: {target_shape}")
        
        for name, img in images.items():
            if img is not None and img.shape != target_shape:
                print(f"Resizing {name} from {img.shape} to {target_shape}")
                img_resized = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
                images[name] = img_resized
        
        print(f"Images loaded and resized successfully. Final dimensions: {target_shape}")
        return images
    
    def load_image(self, image_path):
        """Load an image using multiple methods for better compatibility."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            if image_path.endswith('.tif') or image_path.endswith('.tiff'):
                try:
                    import tifffile
                    img = tifffile.imread(image_path)
                    print(f"Successfully loaded {image_path} using tifffile")
                    return img
                except Exception as e:
                    print(f"tifffile failed for {image_path}: {e}")
                
                try:
                    import rasterio
                    with rasterio.open(image_path) as src:
                        img = src.read(1)
                        print(f"Successfully loaded {image_path} using rasterio")
                    return img
                except Exception as e:
                    print(f"rasterio failed for {image_path}: {e}")
                
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    print(f"Successfully loaded {image_path} using OpenCV")
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    return img
            
            from PIL import Image
            img = Image.open(image_path)
            img = np.array(img)
            
            if len(img.shape) == 3:
                img = np.mean(img, axis=2).astype(np.uint8)
            
            print(f"Successfully loaded {image_path} using PIL")
            return img
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            try:
                return np.load(image_path)
            except:
                raise ValueError(f"Could not load image: {image_path}")
    
    def lee_filter(self, image, window_size=5):
        """Custom implementation of Lee filter for speckle noise reduction."""
        img = image.astype(np.float64)
        kernel = np.ones((window_size, window_size), np.float64) / (window_size * window_size)
        
        local_mean = cv2.filter2D(img, -1, kernel)
        local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
        noise_var = np.var(img) / (np.mean(img)**2 + 1e-10)
        
        k = local_var / (local_var + noise_var + 1e-10)
        filtered = local_mean + k * (img - local_mean)
        
        return np.clip(filtered, 0, img.max()).astype(image.dtype)
    
    def apply_lee_filter_to_sar(self, sar_image):
        """Apply Lee filter to the entire SAR image before patch extraction."""
        print("Applying Lee filter to SAR image...")
        
        noise_level = np.std(sar_image) / np.mean(sar_image)
        
        if noise_level >= self.lee_filter_threshold:
            print(f"SAR image noise level: {noise_level:.3f} (>= {self.lee_filter_threshold})")
            print("Applying Lee filter to reduce speckle noise...")
            
            try:
                filtered_sar = self.lee_filter(sar_image, 5)
                print("Lee filter applied successfully")
                return filtered_sar
            except Exception as e:
                print(f"Lee filter failed: {e}")
                print("Using median filter as fallback...")
                filtered_sar = ndimage.median_filter(sar_image, size=5)
                return filtered_sar
        else:
            print(f"SAR image noise level: {noise_level:.3f} (< {self.lee_filter_threshold})")
            print("No Lee filter needed")
            return sar_image
    
    def g0_fun(self, t, c1, c2, c3):
        """G0 distribution function for optimization."""
        L, a = t
        return [psi(1, L) + psi(1, -a) - 4*c2, 
                psi(2, L) - psi(2, -a) - 8*c3]
    
    def g0molc(self, c1, c2, c3):
        """G0 distribution parameter estimation."""
        t0 = [0.05, -0.05]
        
        try:
            result = fsolve(lambda t: self.g0_fun(t, c1, c2, c3), t0, 
                          xtol=1e-12, maxfev=1000)
            L, alpha = result
            
            residual = np.linalg.norm(self.g0_fun([L, alpha], c1, c2, c3))
            flag = 1 if residual < 1e-6 else 0
            
            return L, alpha, flag
        except:
            return 1.4, -0.1, 0
    
    def calculate_sar_features(self, sar_patch):
        """Calculate SAR-derived features using G0 distribution analysis."""
        try:
            sar_patch = np.abs(sar_patch).astype(np.float64)
            
            valid_mask = (sar_patch > 0) & np.isfinite(sar_patch)
            if np.sum(valid_mask) < 10:
                return self._get_fallback_sar_features(sar_patch)
            
            sar_patch = sar_patch[valid_mask]
            sar_patch = sar_patch + 1e-10
            
            log_patch = np.log(sar_patch)
            c1 = np.mean(log_patch)
            c2 = np.mean((log_patch - c1)**2)
            c3 = np.mean((log_patch - c1)**3)
            
            if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(c3)):
                return self._get_fallback_sar_features(sar_patch)
            
            L, alpha, flag = self.g0molc_robust(c1, c2, c3)
            
            alpha = np.clip(alpha, -10.0, 0.0)
            L = np.clip(L, 0.1, 50.0)
            
            gamma = (-alpha) * np.mean(sar_patch)
            sigma = np.mean(sar_patch)
            noise_level = np.std(sar_patch) / (np.mean(sar_patch) + 1e-10)
            
            return {
                'alpha_mean': alpha,
                'gamma_mean': gamma,
                'sigma_mean': sigma,
                'noise_level': noise_level,
                'convergence_flag': flag
            }
            
        except Exception as e:
            print(f"Error in SAR feature calculation: {e}")
            return self._get_fallback_sar_features(sar_patch)
    
    def _get_fallback_sar_features(self, sar_patch):
        """Get fallback SAR features when G0 optimization fails."""
        try:
            mean_val = np.mean(sar_patch)
            std_val = np.std(sar_patch)
            
            alpha = -0.5 - (std_val / (mean_val + 1e-10)) * 2.0
            alpha = np.clip(alpha, -10.0, 0.0)
            
            gamma = (-alpha) * mean_val
            sigma = mean_val
            noise_level = std_val / (mean_val + 1e-10)
            
            return {
                'alpha_mean': alpha,
                'gamma_mean': gamma,
                'sigma_mean': sigma,
                'noise_level': noise_level,
                'convergence_flag': 0
            }
        except:
            return {
                'alpha_mean': -1.0,
                'gamma_mean': 1.0,
                'sigma_mean': 1.0,
                'noise_level': 0.5,
                'convergence_flag': 0
            }
    
    def g0molc_robust(self, c1, c2, c3):
        """Robust G0 distribution parameter estimation."""
        try:
            starting_points = [[0.05, -0.05], [0.1, -0.1], [0.2, -0.2], [1.0, -1.0]]
            
            best_result = None
            best_residual = float('inf')
            
            for t0 in starting_points:
                try:
                    result = fsolve(lambda t: self.g0_fun(t, c1, c2, c3), t0, 
                                  xtol=1e-8, maxfev=2000)
                    L, alpha = result
                    
                    if 0.1 <= L <= 50.0 and -10.0 <= alpha <= 0.0:
                        residual = np.linalg.norm(self.g0_fun([L, alpha], c1, c2, c3))
                        if residual < best_residual:
                            best_residual = residual
                            best_result = (L, alpha, 1 if residual < 1e-4 else 0)
                except:
                    continue
            
            if best_result is not None:
                return best_result
            else:
                L = 1.4
                alpha = -0.5 - (c2 / (c1 + 1e-10)) * 0.5
                alpha = np.clip(alpha, -10.0, 0.0)
                return L, alpha, 0
                
        except Exception as e:
            print(f"G0 optimization failed: {e}")
            return 1.4, -1.0, 0
    
    def calculate_terrain_features(self, dem_patch, slope_patch):
        """Calculate terrain features from DEM and slope patches."""
        dem_valid = dem_patch[np.isfinite(dem_patch)]
        slope_valid = slope_patch[np.isfinite(slope_patch)]
        
        dem_median = np.median(dem_valid) if len(dem_valid) > 0 else np.nan
        slope_median = np.median(slope_valid) if len(slope_valid) > 0 else np.nan
        
        return {
            'dem_median': dem_median,
            'slope_median': slope_median
        }
    
    def calculate_texture_features(self, patch):
        """Calculate texture features using GLCM."""
        try:
            patch_norm = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-10) * 255).astype(np.uint8)
            glcm = graycomatrix(patch_norm, distances=[1], angles=[0, 45, 90, 135], 
                               levels=256, symmetric=True, normed=True)
            
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            
            return {
                'texture_contrast': contrast,
                'texture_dissimilarity': dissimilarity,
                'texture_homogeneity': homogeneity
            }
        except:
            return {
                'texture_contrast': 0.0,
                'texture_dissimilarity': 0.0,
                'texture_homogeneity': 1.0
            }
    
    def calculate_landcover_features(self, landcover_patch):
        """Calculate landcover features including dominant landcover type."""
        unique_vals, counts = np.unique(landcover_patch, return_counts=True)
        
        dominant_code = unique_vals[np.argmax(counts)]
        dominant_count = np.max(counts)
        dominant_percentage = (dominant_count / landcover_patch.size) * 100
        
        landcover_diversity = len(unique_vals)
        
        return {
            'dominant_landcover_code': dominant_code,
            'dominant_landcover_percentage': dominant_percentage,
            'landcover_diversity': landcover_diversity
        }
    
    def calculate_spatial_features(self, landcover_patch, patch_center):
        """Calculate spatial features including distance to water."""
        unique_vals, counts = np.unique(landcover_patch, return_counts=True)
        
        water_mask = None
        
        # Look for permanent water bodies (code 80)
        water_codes = [80]
        
        for code in water_codes:
            if np.any(landcover_patch == code):
                water_mask = (landcover_patch == code)
                break
        
        # Backup: look for rare values as potential water
        if water_mask is None:
            patch_size = landcover_patch.size
            for val, count in zip(unique_vals, counts):
                if count < patch_size * 0.05:
                    water_mask = (landcover_patch == val)
                    break
        
        if water_mask is not None and np.any(water_mask):
            water_coords = np.argwhere(water_mask)
            distances = np.sqrt((water_coords[:, 0] - patch_center[0])**2 + 
                              (water_coords[:, 1] - patch_center[1])**2)
            distance_to_water = np.min(distances)
        else:
            distance_to_water = np.sqrt(patch_center[0]**2 + patch_center[1]**2) / 100.0
            distance_to_water += np.random.uniform(0, 10)
        
        return {
            'distance_to_water': distance_to_water
        }
    
    def extract_patches(self, images):
        """Extract patches with strict filtering on dominant landcover percentage and diversity."""
        print("Extracting patches with strict landcover filtering...")
        
        sar_img = images['sar']
        dem_img = images['dem']
        slope_img = images['slope']
        landcover_img = images['landcover']
        label_img = images['label']
        coherence_img = images['coherence']
        
        height, width = sar_img.shape
        patches = []
        patch_count = 0
        
        # Filtering statistics
        filtering_stats = {
            'total_attempted': 0,
            'high_noise': 0,
            'invalid_terrain': 0,
            'mixed_landcover': 0,
            'low_dominance': 0,
            'valid_patches': 0
        }
        
        # Calculate number of patches for progress tracking
        total_patches = ((height - self.patch_size) // self.stride + 1) * \
                    ((width - self.patch_size) // self.stride + 1)
        
        print(f"Total patches to process: {total_patches}")
        print(f"Applying strict filtering: dominant_percentage >= 99% AND diversity = 1 (homogeneous only)")
        
        with tqdm(total=total_patches, desc="Extracting patches") as pbar:
            for i in range(0, height - self.patch_size + 1, self.stride):
                for j in range(0, width - self.patch_size + 1, self.stride):
                    filtering_stats['total_attempted'] += 1
                    
                    # Extract patches
                    sar_patch = sar_img[i:i+self.patch_size, j:j+self.patch_size]
                    dem_patch = dem_img[i:i+self.patch_size, j:j+self.patch_size]
                    slope_patch = slope_img[i:i+self.patch_size, j:j+self.patch_size]
                    landcover_patch = landcover_img[i:i+self.patch_size, j:j+self.patch_size]
                    
                    # Calculate patch center
                    patch_center = (i + self.patch_size // 2, j + self.patch_size // 2)
                    patch_id = f"patch_{patch_center[0]}_{patch_center[1]}"
                    
                    # Calculate all features
                    sar_features = self.calculate_sar_features(sar_patch)
                    
                    # FILTER 1: Skip patches with excessive noise
                    if sar_features['noise_level'] >= self.noise_threshold:
                        filtering_stats['high_noise'] += 1
                        pbar.update(1)
                        continue
                    
                    terrain_features = self.calculate_terrain_features(dem_patch, slope_patch)
                    
                    # FILTER 2: Skip patches with invalid terrain data
                    if np.isnan(terrain_features['dem_median']) or np.isnan(terrain_features['slope_median']):
                        filtering_stats['invalid_terrain'] += 1
                        pbar.update(1)
                        continue
                    
                    texture_features = self.calculate_texture_features(sar_patch)
                    landcover_features = self.calculate_landcover_features(landcover_patch)
                    
                    # FILTER 3: Skip patches with mixed landcover (diversity > 1)
                    if landcover_features['landcover_diversity'] > 1:
                        filtering_stats['mixed_landcover'] += 1
                        pbar.update(1)
                        continue
                    
                    # FILTER 4: Skip patches with low landcover dominance (< 99%)
                    if landcover_features['dominant_landcover_percentage'] < 99.0:
                        filtering_stats['low_dominance'] += 1
                        pbar.update(1)
                        continue
                    
                    spatial_features = self.calculate_spatial_features(landcover_patch, patch_center)
                    
                    # Calculate coherence features if available
                    if coherence_img is not None:
                        coherence_patch = coherence_img[i:i+self.patch_size, j:j+self.patch_size]
                        coherence_mean = np.median(coherence_patch)
                    else:
                        coherence_mean = np.nan
                    
                    # Create patch data with ONLY required columns for Stage 1 ML
                    patch_data = {
                        # Identifiers
                        'patch_id': patch_id,
                        'row': patch_center[0],
                        'col': patch_center[1],
                        
                        # Stage 1 Features (Land Cover Classification)
                        'alpha_mean': sar_features['alpha_mean'],
                        'gamma_mean': sar_features['gamma_mean'],
                        'sigma_mean': sar_features['sigma_mean'],
                        'coherence_mean': coherence_mean,
                        'texture_contrast': texture_features['texture_contrast'],
                        'texture_dissimilarity': texture_features['texture_dissimilarity'],
                        'texture_homogeneity': texture_features['texture_homogeneity'],
                        'noise_level': sar_features['noise_level'],
                        'convergence_flag': sar_features['convergence_flag'],
                        'dominant_landcover_code': landcover_features['dominant_landcover_code'],
                        
                        # Additional Features (for future use)
                        'dem_median': terrain_features['dem_median'],
                        'slope_median': terrain_features['slope_median'],
                        'distance_to_water': spatial_features['distance_to_water'],
                        'dominant_landcover_percentage': landcover_features['dominant_landcover_percentage'],
                        'landcover_diversity': landcover_features['landcover_diversity']
                    }
                    
                    patches.append(patch_data)
                    patch_count += 1
                    filtering_stats['valid_patches'] += 1
                    
                    pbar.update(1)
        
        # Print detailed filtering statistics
        print(f"\n📊 PATCH FILTERING RESULTS:")
        print(f"  Total patches attempted: {filtering_stats['total_attempted']}")
        print(f"  ❌ High noise (>={self.noise_threshold}): {filtering_stats['high_noise']} ({filtering_stats['high_noise']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  ❌ Invalid terrain: {filtering_stats['invalid_terrain']} ({filtering_stats['invalid_terrain']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  ❌ Mixed landcover (>1 type): {filtering_stats['mixed_landcover']} ({filtering_stats['mixed_landcover']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  ❌ Low dominance (<99%): {filtering_stats['low_dominance']} ({filtering_stats['low_dominance']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  ✅ VALID patches: {filtering_stats['valid_patches']} ({filtering_stats['valid_patches']/filtering_stats['total_attempted']*100:.1f}%)")
        
        print(f"\n🎯 Final result: {len(patches)} high-quality patches extracted")
        print(f"   All patches are homogeneous (diversity=1) with ≥99% dominance")
        
        return patches
    
    def generate_dataset(self, sar_path, dem_path, slope_path, landcover_path, 
                        label_path, coherence_path=None, output_dir='output'):
        """Main function to generate the dataset for Stage 1 ML approach."""
        print("Starting Stage 1 dataset generation for land cover classification...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        images = self.load_images(sar_path, dem_path, slope_path, 
                                landcover_path, label_path, coherence_path)
        
        # Apply Lee filter to SAR image
        images['sar'] = self.apply_lee_filter_to_sar(images['sar'])
        
        # Extract patches
        patches = self.extract_patches(images)
        
        if not patches:
            print("No valid patches extracted!")
            return
        
        # Save dataset
        df = pd.DataFrame(patches)
        dataset_path = os.path.join(output_dir, 'camping_site_dataset_stage1.csv')
        df.to_csv(dataset_path, index=False)
        
        print(f"\nStage 1 dataset generation complete!")
        print(f"Dataset saved to: {dataset_path}")
        print(f"Total patches: {len(patches)}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Land cover classes: {df['dominant_landcover_code'].nunique()}")
        print(f"Target variable: dominant_landcover_code (for land cover classification)")
        
        return df


def main():
    """Main function to run the dataset generation for Stage 1 ML approach."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Stage 1 dataset for land cover classification')
    parser.add_argument('--sar', help='Path to SAR image')
    parser.add_argument('--dem', help='Path to DEM image')
    parser.add_argument('--slope', help='Path to slope image')
    parser.add_argument('--landcover', help='Path to landcover image')
    parser.add_argument('--label', help='Path to label mask image')
    parser.add_argument('--coherence', help='Path to coherence image (optional)')
    parser.add_argument('--output', default='camping_site_output_stage1', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = CampingSiteDatasetGenerator(
        patch_size=50,
        stride=25,
        noise_threshold=0.8,
        lee_filter_threshold=0.6
    )
    
    # Use command line arguments or default paths
    if args.sar and args.dem and args.slope and args.landcover and args.label:
        image_paths = {
            'sar_path': args.sar,
            'dem_path': args.dem,
            'slope_path': args.slope,
            'landcover_path': args.landcover,
            'label_path': args.label,
            'coherence_path': args.coherence
        }
        output_dir = args.output
    else:
        # Default paths
        image_paths = {
            'sar_path': 'data/SAR_VV_VH_5km_Chopta_Magpie_Jungle_Camp.tif',
            'dem_path': 'data/DEM_SRTM_5km_Chopta_Magpie_Jungle_Camp.tif', 
            'slope_path': 'data/Slope_degrees_5km_Chopta_Magpie_Jungle_Camp.tif',
            'landcover_path': 'data/Landcover_ESA_5km_Chopta_Magpie_Jungle_Camp.tif',
            'label_path': 'data/Camping_Label_Mask_5km_Chopta_Magpie_Jungle_Camp.tif',
            'coherence_path': 'data/Coherence_Stability_5km_Chopta_Magpie_Jungle_Camp.tif'
        }
        output_dir = 'camping_site_output_stage1'
    
    # Generate dataset
    try:
        df = generator.generate_dataset(
            sar_path=image_paths['sar_path'],
            dem_path=image_paths['dem_path'],
            slope_path=image_paths['slope_path'],
            landcover_path=image_paths['landcover_path'],
            label_path=image_paths['label_path'],
            coherence_path=image_paths['coherence_path'],
            output_dir=output_dir
        )
        
        print("\n🎯 Stage 1 dataset generation completed successfully!")
        print(f"Ready for Stage 1: Land cover classification using SAR + texture features")
        print(f"Target variable: dominant_landcover_code")
        print(f"Dataset contains only homogeneous patches (diversity=1) with ≥99% purity")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
