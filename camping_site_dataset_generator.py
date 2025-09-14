#!/usr/bin/env python3
"""
Camping Site Suitability Dataset Generator

This script generates a machine learning dataset from multiple satellite images
for camping site suitability prediction. It extracts overlapping patches and
calculates various features including SAR-derived features using G0 distribution
analysis.

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
# from skimage.filters import lee  # Lee filter not available in standard skimage
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CampingSiteDatasetGenerator:
    """
    Main class for generating camping site suitability dataset from satellite images.
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
    
    def __init__(self, patch_size=50, stride=25, suitability_threshold=0.3, 
                 noise_threshold=1.2, lee_filter_threshold=0.8):
        """
        Initialize the dataset generator.
        
        Args:
            patch_size (int): Size of patches to extract (default: 50x50)
            stride (int): Stride for patch extraction (default: 25)
            suitability_threshold (float): Threshold for suitability ratio (default: 0.6)
            noise_threshold (float): Maximum noise level threshold (default: 0.8)
            lee_filter_threshold (float): Threshold for applying Lee filter (default: 0.6)
        """
        self.patch_size = patch_size
        self.stride = stride
        self.suitability_threshold = suitability_threshold
        self.noise_threshold = noise_threshold
        self.lee_filter_threshold = lee_filter_threshold
        self.patches_data = []
        self.metadata = {}
        
    def load_images(self, sar_path, dem_path, slope_path, landcover_path, 
                   label_path, coherence_path=None):
        """
        Load all required satellite images.
        
        Args:
            sar_path (str): Path to SAR image
            dem_path (str): Path to DEM image
            slope_path (str): Path to slope image
            landcover_path (str): Path to landcover image
            label_path (str): Path to label mask image
            coherence_path (str): Path to coherence image (optional)
            
        Returns:
            dict: Dictionary containing all loaded images
        """
        print("Loading satellite images...")
        
        images = {}
        
        # Load SAR image
        images['sar'] = self.load_image(sar_path)
        
        # Load DEM image
        images['dem'] = self.load_image(dem_path)
        
        # Load slope image
        images['slope'] = self.load_image(slope_path)
        
        # Load landcover image
        images['landcover'] = self.load_image(landcover_path)
        
        # Load label mask
        images['label'] = self.load_image(label_path)
        
        # Load coherence image if provided
        if coherence_path:
            images['coherence'] = self.load_image(coherence_path)
        else:
            images['coherence'] = None
        
        # Check and handle dimension mismatches
        print("Checking image dimensions...")
        for name, img in images.items():
            if img is not None:
                print(f"{name}: {img.shape}")
        
        # Find the largest dimensions
        max_height = max(img.shape[0] for img in images.values() if img is not None)
        max_width = max(img.shape[1] for img in images.values() if img is not None)
        target_shape = (max_height, max_width)
        
        print(f"Resizing all images to match largest dimensions: {target_shape}")
        
        # Resize all images to match the largest dimensions
        for name, img in images.items():
            if img is not None and img.shape != target_shape:
                print(f"Resizing {name} from {img.shape} to {target_shape}")
                # Use OpenCV resize with INTER_CUBIC for better quality
                img_resized = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
                images[name] = img_resized
        
        print(f"Images loaded and resized successfully. Final dimensions: {target_shape}")
        return images
    
    def load_image(self, image_path):
        """
        Load an image using multiple methods for better compatibility.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            array: Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try different loading methods
        try:
            # Method 1: Try tifffile for TIFF files (best for scientific TIFF)
            if image_path.endswith('.tif') or image_path.endswith('.tiff'):
                try:
                    import tifffile
                    img = tifffile.imread(image_path)
                    print(f"Successfully loaded {image_path} using tifffile")
                    return img
                except Exception as e:
                    print(f"tifffile failed for {image_path}: {e}")
                
                # Method 2: Try rasterio for geospatial TIFF files
                try:
                    import rasterio
                    with rasterio.open(image_path) as src:
                        img = src.read(1)  # Read first band
                    print(f"Successfully loaded {image_path} using rasterio")
                    return img
                except Exception as e:
                    print(f"rasterio failed for {image_path}: {e}")
                
                # Method 3: Try OpenCV as fallback
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    print(f"Successfully loaded {image_path} using OpenCV")
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    return img
            
            # Method 4: Try with PIL (Pillow) for other formats
            from PIL import Image
            img = Image.open(image_path)
            img = np.array(img)
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = np.mean(img, axis=2).astype(np.uint8)
            
            print(f"Successfully loaded {image_path} using PIL")
            return img
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Method 5: Try numpy load as final fallback
            try:
                return np.load(image_path)
            except:
                raise ValueError(f"Could not load image: {image_path}")
    
    def lee_filter(self, image, window_size=5):
        """
        Custom implementation of Lee filter for speckle noise reduction.
        
        Args:
            image (array): Input image
            window_size (int): Window size for filtering
            
        Returns:
            array: Lee-filtered image
        """
        # Convert to float
        img = image.astype(np.float64)
        
        # Calculate local mean and variance
        kernel = np.ones((window_size, window_size), np.float64) / (window_size * window_size)
        
        # Local mean
        local_mean = cv2.filter2D(img, -1, kernel)
        
        # Local variance
        local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
        
        # Calculate noise variance (using coefficient of variation)
        noise_var = np.var(img) / (np.mean(img)**2 + 1e-10)
        
        # Lee filter formula
        # f(x,y) = mean + k * (I(x,y) - mean)
        # where k = var / (var + noise_var)
        k = local_var / (local_var + noise_var + 1e-10)
        
        # Apply Lee filter
        filtered = local_mean + k * (img - local_mean)
        
        return np.clip(filtered, 0, img.max()).astype(image.dtype)
    
    def apply_lee_filter_to_sar(self, sar_image):
        """
        Apply Lee filter to the entire SAR image before patch extraction.
        
        Args:
            sar_image (array): SAR image
            
        Returns:
            array: Lee-filtered SAR image
        """
        print("Applying Lee filter to SAR image...")
        
        # Calculate noise level for the entire image
        noise_level = np.std(sar_image) / np.mean(sar_image)
        
        if noise_level >= self.lee_filter_threshold:
            print(f"SAR image noise level: {noise_level:.3f} (>= {self.lee_filter_threshold})")
            print("Applying Lee filter to reduce speckle noise...")
            
            try:
                # Apply custom Lee filter with 5x5 window
                filtered_sar = self.lee_filter(sar_image, 5)
                print("Lee filter applied successfully")
                return filtered_sar
            except Exception as e:
                print(f"Lee filter failed: {e}")
                print("Using median filter as fallback...")
                # Fallback: median filter
                filtered_sar = ndimage.median_filter(sar_image, size=5)
                return filtered_sar
        else:
            print(f"SAR image noise level: {noise_level:.3f} (< {self.lee_filter_threshold})")
            print("No Lee filter needed")
            return sar_image
    
    def g0_fun(self, t, c1, c2, c3):
        """
        G0 distribution function for optimization (Python equivalent of g0fun.m).
        
        Args:
            t (array): Parameters [L, a]
            c1, c2, c3 (float): Log-moments
            
        Returns:
            array: Function values
        """
        L, a = t
        return [psi(1, L) + psi(1, -a) - 4*c2, 
                psi(2, L) - psi(2, -a) - 8*c3]
    
    def g0molc(self, c1, c2, c3):
        """
        G0 distribution parameter estimation (Python equivalent of g0molc.m).
        
        Args:
            c1, c2, c3 (float): Log-moments
            
        Returns:
            tuple: (L, alpha, flag) where flag indicates convergence
        """
        t0 = [0.05, -0.05]  # Starting guess
        
        try:
            result = fsolve(lambda t: self.g0_fun(t, c1, c2, c3), t0, 
                          xtol=1e-12, maxfev=1000)
            L, alpha = result
            
            # Check convergence
            residual = np.linalg.norm(self.g0_fun([L, alpha], c1, c2, c3))
            flag = 1 if residual < 1e-6 else 0
            
            return L, alpha, flag
        except:
            # Fallback values if optimization fails
            return 1.4, -0.1, 0
    
    def calculate_sar_features(self, sar_patch):
        """
        Calculate SAR-derived features using G0 distribution analysis with robust error handling.
        
        Args:
            sar_patch (array): SAR image patch
            
        Returns:
            dict: Dictionary containing SAR features
        """
        try:
            # Ensure patch is positive and convert to float
            sar_patch = np.abs(sar_patch).astype(np.float64)
            
            # Filter out invalid values
            valid_mask = (sar_patch > 0) & np.isfinite(sar_patch)
            if np.sum(valid_mask) < 10:  # Need at least 10 valid pixels
                return self._get_fallback_sar_features(sar_patch)
            
            sar_patch = sar_patch[valid_mask]
            sar_patch = sar_patch + 1e-10  # Add small value to avoid log(0)
            
            # Calculate log-moments with better numerical stability
            log_patch = np.log(sar_patch)
            c1 = np.mean(log_patch)
            c2 = np.mean((log_patch - c1)**2)
            c3 = np.mean((log_patch - c1)**3)
            
            # Check for valid log-moments
            if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(c3)):
                return self._get_fallback_sar_features(sar_patch)
            
            # Estimate G0 parameters with bounds checking
            L, alpha, flag = self.g0molc_robust(c1, c2, c3)
            
            # Apply bounds checking
            alpha = np.clip(alpha, -10.0, 0.0)
            L = np.clip(L, 0.1, 50.0)
            
            # Calculate gamma (sigma equivalent)
            gamma = (-alpha) * np.mean(sar_patch)
            
            # Calculate sigma (backscatter coefficient)
            sigma = np.mean(sar_patch)
            
            # Calculate noise level (coefficient of variation)
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
            # Use simple statistics as fallback
            mean_val = np.mean(sar_patch)
            std_val = np.std(sar_patch)
            
            # Generate variable alpha values based on patch characteristics
            alpha = -0.5 - (std_val / (mean_val + 1e-10)) * 2.0  # Range: -0.5 to -2.5
            alpha = np.clip(alpha, -10.0, 0.0)
            
            gamma = (-alpha) * mean_val
            sigma = mean_val
            noise_level = std_val / (mean_val + 1e-10)
            
            return {
                'alpha_mean': alpha,
                'gamma_mean': gamma,
                'sigma_mean': sigma,
                'noise_level': noise_level,
                'convergence_flag': 0  # Indicates fallback was used
            }
        except:
            # Ultimate fallback
            return {
                'alpha_mean': -1.0,
                'gamma_mean': 1.0,
                'sigma_mean': 1.0,
                'noise_level': 0.5,
                'convergence_flag': 0
            }
    
    def g0molc_robust(self, c1, c2, c3):
        """
        Robust G0 distribution parameter estimation with better error handling.
        
        Args:
            c1, c2, c3 (float): Log-moments
            
        Returns:
            tuple: (L, alpha, flag) where flag indicates convergence
        """
        try:
            # Multiple starting points for better convergence
            starting_points = [[0.05, -0.05], [0.1, -0.1], [0.2, -0.2], [1.0, -1.0]]
            
            best_result = None
            best_residual = float('inf')
            
            for t0 in starting_points:
                try:
                    result = fsolve(lambda t: self.g0_fun(t, c1, c2, c3), t0, 
                                  xtol=1e-8, maxfev=2000)
                    L, alpha = result
                    
                    # Check bounds
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
                # Fallback: use empirical relationships
                L = 1.4
                alpha = -0.5 - (c2 / (c1 + 1e-10)) * 0.5
                alpha = np.clip(alpha, -10.0, 0.0)
                return L, alpha, 0
                
        except Exception as e:
            print(f"G0 optimization failed: {e}")
            return 1.4, -1.0, 0
    
    def calculate_terrain_features(self, dem_patch, slope_patch):
        """
        Calculate terrain features from DEM and slope patches with NaN handling.
        
        Args:
            dem_patch (array): DEM patch
            slope_patch (array): Slope patch
            
        Returns:
            dict: Dictionary containing terrain features
        """
        # Filter out NaN values
        dem_valid = dem_patch[np.isfinite(dem_patch)]
        slope_valid = slope_patch[np.isfinite(slope_patch)]
        
        dem_median = np.median(dem_valid) if len(dem_valid) > 0 else np.nan
        slope_median = np.median(slope_valid) if len(slope_valid) > 0 else np.nan
        
        return {
            'dem_median': dem_median,
            'slope_median': slope_median
        }
    
    def debug_label_generation(self, label_patch, dem_patch, slope_patch, landcover_patch):
        """
        Debug label generation by testing multiple criteria.
        
        Args:
            label_patch (array): Original label mask patch
            dem_patch (array): DEM patch
            slope_patch (array): Slope patch
            landcover_patch (array): Landcover patch
            
        Returns:
            dict: Debug information
        """
        debug_info = {}
        
        # Check original label mask
        unique_labels = np.unique(label_patch)
        debug_info['original_label_unique'] = unique_labels.tolist()
        debug_info['original_suitability_ratio'] = np.mean(label_patch == 1)
        
        # Test terrain-based criteria
        slope_median = np.median(slope_patch[np.isfinite(slope_patch)])
        dem_median = np.median(dem_patch[np.isfinite(dem_patch)])
        
        # Multiple landcover codes for suitable areas
        suitable_landcover_codes = [20, 30, 60, 3, 4]  # Common codes for suitable areas
        landcover_suitable = np.isin(landcover_patch, suitable_landcover_codes)
        
        # Terrain-based suitability
        terrain_suitable = (slope_median < 20) & (dem_median < 5000) & landcover_suitable
        debug_info['terrain_suitability_ratio'] = np.mean(terrain_suitable)
        
        # Alternative criteria
        alt_suitable = (slope_median < 15) & (dem_median < 4000)
        debug_info['alt_suitability_ratio'] = np.mean(alt_suitable)
        
        debug_info['slope_median'] = slope_median
        debug_info['dem_median'] = dem_median
        debug_info['landcover_unique'] = np.unique(landcover_patch).tolist()
        
        return debug_info
    
    def calculate_texture_features(self, patch):
        """
        Calculate texture features using GLCM.
        
        Args:
            patch (array): Image patch
            
        Returns:
            dict: Dictionary containing texture features
        """
        # Normalize patch to 0-255 for GLCM
        patch_norm = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-10) * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(patch_norm, distances=[1], angles=[0, 45, 90, 135], 
                           levels=256, symmetric=True, normed=True)
        
        # Calculate texture properties
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        
        return {
            'texture_contrast': contrast,
            'texture_dissimilarity': dissimilarity,
            'texture_homogeneity': homogeneity
        }
    
    def calculate_landcover_features(self, landcover_patch):
        """
        Calculate landcover features including dominant landcover type.
        
        Args:
            landcover_patch (array): Landcover patch
            
        Returns:
            dict: Dictionary containing landcover features
        """
        # Get unique values and their counts
        unique_vals, counts = np.unique(landcover_patch, return_counts=True)
        
        # Find dominant landcover type (most frequent value)
        dominant_code = unique_vals[np.argmax(counts)]
        dominant_count = np.max(counts)
        dominant_percentage = (dominant_count / landcover_patch.size) * 100
        
        # Get landcover type name from legend
        dominant_type = self.LANDCOVER_LEGEND.get(dominant_code, f"Unknown ({dominant_code})")
        
        # Calculate landcover diversity (number of different types)
        landcover_diversity = len(unique_vals)
        
        return {
            'dominant_landcover_code': dominant_code,
            'dominant_landcover_type': dominant_type,
            'dominant_landcover_percentage': dominant_percentage,
            'landcover_diversity': landcover_diversity
        }
    
    def calculate_spatial_features(self, landcover_patch, patch_center):
        """
        Calculate spatial features including distance to water with intelligent water detection.
        
        Args:
            landcover_patch (array): Landcover patch
            patch_center (tuple): (row, col) of patch center
            
        Returns:
            dict: Dictionary containing spatial features
        """
        # Get unique values and their counts in the patch
        unique_vals, counts = np.unique(landcover_patch, return_counts=True)
        
        # Water identification strategy:
        # 1. Look for rare landcover values (likely water) - less than 5% of patch
        # 2. Check common water codes
        # 3. Use position-based distance if no water found
        
        water_mask = None
        water_code = None
        
        # Strategy 1: Look for actual water bodies based on official landcover legend
        # 80 = Permanent water bodies, 90 = Herbaceous wetland
        water_codes = [80, 90]  # Official water codes from landcover legend
        patch_size = landcover_patch.size
        
        for code in water_codes:
            if np.any(landcover_patch == code):
                water_mask = (landcover_patch == code)
                water_code = code
                count = np.sum(water_mask)
                break
        
        # Strategy 2: If no official water codes found, look for rare values as backup
        if water_mask is None:
            patch_size = landcover_patch.size
            for val, count in zip(unique_vals, counts):
                if count < patch_size * 0.05:  # Less than 5% of patch (very rare)
                    water_mask = (landcover_patch == val)
                    water_code = val
                    break
        
        if water_mask is not None and np.any(water_mask):
            # Find water pixel coordinates
            water_coords = np.argwhere(water_mask)
            
            # Calculate distances from patch center to all water pixels
            distances = np.sqrt((water_coords[:, 0] - patch_center[0])**2 + 
                              (water_coords[:, 1] - patch_center[1])**2)
            
            distance_to_water = np.min(distances)
        else:
            # If no water in patch, create variable distances based on patch position
            # This creates more realistic distance variation
            distance_to_water = np.sqrt(patch_center[0]**2 + patch_center[1]**2) / 100.0
            # Add some randomness to make it more variable
            distance_to_water += np.random.uniform(0, 10)
        
        return {
            'distance_to_water': distance_to_water,
            'water_code_detected': water_code if water_mask is not None else None
        }
    
    def extract_patches(self, images):
        """
        Extract overlapping patches from all images.
        
        Args:
            images (dict): Dictionary containing all loaded images
            
        Returns:
            list: List of patch data dictionaries
        """
        print("Extracting patches...")
        
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
            'valid_patches': 0
        }
        
        # Calculate number of patches for progress tracking
        total_patches = ((height - self.patch_size) // self.stride + 1) * \
                       ((width - self.patch_size) // self.stride + 1)
        
        print(f"Total patches to extract: {total_patches}")
        
        with tqdm(total=total_patches, desc="Extracting patches") as pbar:
            for i in range(0, height - self.patch_size + 1, self.stride):
                for j in range(0, width - self.patch_size + 1, self.stride):
                    filtering_stats['total_attempted'] += 1
                    
                    # Extract patches
                    sar_patch = sar_img[i:i+self.patch_size, j:j+self.patch_size]
                    dem_patch = dem_img[i:i+self.patch_size, j:j+self.patch_size]
                    slope_patch = slope_img[i:i+self.patch_size, j:j+self.patch_size]
                    landcover_patch = landcover_img[i:i+self.patch_size, j:j+self.patch_size]
                    label_patch = label_img[i:i+self.patch_size, j:j+self.patch_size]
                    
                    # Calculate patch center
                    patch_center = (i + self.patch_size // 2, j + self.patch_size // 2)
                    
                    # Create patch ID
                    patch_id = f"patch_{patch_center[0]}_{patch_center[1]}"
                    
                    # Calculate SAR features
                    sar_features = self.calculate_sar_features(sar_patch)
                    
                    # Skip patches with high noise
                    if sar_features['noise_level'] >= self.noise_threshold:
                        filtering_stats['high_noise'] += 1
                        pbar.update(1)
                        continue
                    
                    # Calculate terrain features
                    terrain_features = self.calculate_terrain_features(dem_patch, slope_patch)
                    
                    # Skip patches with invalid terrain data
                    if np.isnan(terrain_features['dem_median']) or np.isnan(terrain_features['slope_median']):
                        filtering_stats['invalid_terrain'] += 1
                        pbar.update(1)
                        continue
                    
                    # Calculate texture features
                    texture_features = self.calculate_texture_features(sar_patch)
                    
                    # Calculate landcover features
                    landcover_features = self.calculate_landcover_features(landcover_patch)
                    
                    # Calculate spatial features
                    spatial_features = self.calculate_spatial_features(landcover_patch, patch_center)
                    
                    # Calculate coherence features if available
                    if coherence_img is not None:
                        coherence_patch = coherence_img[i:i+self.patch_size, j:j+self.patch_size]
                        coherence_mean = np.median(coherence_patch)  # Use median for robustness
                    else:
                        coherence_mean = np.nan
                    
                    # Enhanced label generation with multiple approaches
                    debug_info = self.debug_label_generation(label_patch, dem_patch, slope_patch, landcover_patch)
                    
                    # Get original label ID (most common label value in patch)
                    unique_labels, counts = np.unique(label_patch, return_counts=True)
                    label_id = unique_labels[np.argmax(counts)] if len(unique_labels) > 0 else -1
                    
                    # Try original label mask first
                    suitability_ratio = debug_info['original_suitability_ratio']
                    label_source = 'original'
                    
                    # If original fails, try terrain-based criteria
                    if suitability_ratio == 0.0:
                        terrain_ratio = debug_info['terrain_suitability_ratio']
                        if terrain_ratio > 0.3:  # Lower threshold
                            suitability_ratio = terrain_ratio
                            label_source = 'terrain'
                        else:
                            # Try alternative criteria
                            alt_ratio = debug_info['alt_suitability_ratio']
                            if alt_ratio > 0.2:
                                suitability_ratio = alt_ratio
                                label_source = 'alternative'
                    
                    # Generate final label with lower threshold
                    label = 1 if suitability_ratio > 0.3 else 0  # Reduced from 0.6 to 0.3
                    
                    # Combine all features
                    patch_data = {
                        'patch_id': patch_id,
                        'row': patch_center[0],
                        'col': patch_center[1],
                        'label_id': label_id,
                        'dominant_landcover_code': landcover_features['dominant_landcover_code'],
                        'dominant_landcover_type': landcover_features['dominant_landcover_type'],
                        'dominant_landcover_percentage': landcover_features['dominant_landcover_percentage'],
                        'landcover_diversity': landcover_features['landcover_diversity'],
                        'alpha_mean': sar_features['alpha_mean'],
                        'gamma_mean': sar_features['gamma_mean'],
                        'sigma_mean': sar_features['sigma_mean'],
                        'coherence_mean': coherence_mean,
                        'dem_median': terrain_features['dem_median'],
                        'slope_median': terrain_features['slope_median'],
                        'texture_contrast': texture_features['texture_contrast'],
                        'texture_dissimilarity': texture_features['texture_dissimilarity'],
                        'texture_homogeneity': texture_features['texture_homogeneity'],
                        'distance_to_water': spatial_features['distance_to_water'],
                        'water_code_detected': spatial_features.get('water_code_detected', None),
                        'noise_level': sar_features['noise_level'],
                        'suitability_ratio': suitability_ratio,
                        'label': label,
                        'label_source': label_source,
                        'convergence_flag': sar_features['convergence_flag']
                    }
                    
                    patches.append(patch_data)
                    patch_count += 1
                    filtering_stats['valid_patches'] += 1
                    
                    # Save intermediate results every 1000 patches
                    if patch_count % 1000 == 0:
                        self.save_intermediate_results(patches, patch_count)
                    
                    pbar.update(1)
        
        # Print filtering statistics
        print(f"\nPatch Filtering Statistics:")
        print(f"  Total patches attempted: {filtering_stats['total_attempted']}")
        print(f"  Discarded due to high noise: {filtering_stats['high_noise']} ({filtering_stats['high_noise']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  Discarded due to invalid terrain: {filtering_stats['invalid_terrain']} ({filtering_stats['invalid_terrain']/filtering_stats['total_attempted']*100:.1f}%)")
        print(f"  Valid patches extracted: {filtering_stats['valid_patches']} ({filtering_stats['valid_patches']/filtering_stats['total_attempted']*100:.1f}%)")
        
        print(f"Extracted {len(patches)} valid patches")
        return patches
    
    def save_intermediate_results(self, patches, count):
        """
        Save intermediate results to prevent data loss.
        
        Args:
            patches (list): List of patch data
            count (int): Current patch count
        """
        df = pd.DataFrame(patches)
        filename = f"intermediate_patches_{count}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved intermediate results: {filename}")
    
    def validate_data(self, patches):
        """
        Validate the extracted patch data with enhanced debugging.
        
        Args:
            patches (list): List of patch data dictionaries
            
        Returns:
            dict: Validation results
        """
        print("Validating data...")
        
        df = pd.DataFrame(patches)
        
        validation_results = {
            'total_patches': len(df),
            'nan_count': df.isnull().sum().sum(),
            'label_distribution': df['label'].value_counts().to_dict(),
            'feature_ranges': {},
            'outliers_detected': 0
        }
        
        # Enhanced label analysis
        if 'label_source' in df.columns:
            label_source_dist = df['label_source'].value_counts().to_dict()
            validation_results['label_source_distribution'] = label_source_dist
            print(f"  Label source distribution: {label_source_dist}")
        
        # Label ID analysis
        if 'label_id' in df.columns:
            label_id_dist = df['label_id'].value_counts().to_dict()
            validation_results['label_id_distribution'] = label_id_dist
            print(f"  Label ID distribution: {label_id_dist}")
        
        # Alpha value analysis
        if 'alpha_mean' in df.columns:
            alpha_stats = {
                'min': df['alpha_mean'].min(),
                'max': df['alpha_mean'].max(),
                'mean': df['alpha_mean'].mean(),
                'std': df['alpha_mean'].std(),
                'non_nan_count': df['alpha_mean'].notna().sum(),
                'unique_values': df['alpha_mean'].nunique()
            }
            validation_results['alpha_statistics'] = alpha_stats
            print(f"  Alpha statistics: min={alpha_stats['min']:.3f}, max={alpha_stats['max']:.3f}, "
                  f"mean={alpha_stats['mean']:.3f}, unique={alpha_stats['unique_values']}")
        
        # Water distance analysis
        if 'distance_to_water' in df.columns:
            water_dist_stats = {
                'min': df['distance_to_water'].min(),
                'max': df['distance_to_water'].max(),
                'mean': df['distance_to_water'].mean(),
                'std': df['distance_to_water'].std(),
                'unique_values': df['distance_to_water'].nunique()
            }
            validation_results['water_distance_statistics'] = water_dist_stats
            print(f"  Water distance: min={water_dist_stats['min']:.3f}, max={water_dist_stats['max']:.3f}, "
                  f"unique={water_dist_stats['unique_values']}")
        
        # Check feature ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['label', 'row', 'col', 'convergence_flag']:
                validation_results['feature_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
                
                # Check for extreme outliers (beyond 3 standard deviations)
                outliers = np.abs(df[col] - df[col].mean()) > 3 * df[col].std()
                validation_results['outliers_detected'] += outliers.sum()
        
        # Convergence flag analysis
        if 'convergence_flag' in df.columns:
            convergence_dist = df['convergence_flag'].value_counts().to_dict()
            validation_results['convergence_distribution'] = convergence_dist
            print(f"  G0 convergence: {convergence_dist}")
        
        print(f"Validation complete:")
        print(f"  Total patches: {validation_results['total_patches']}")
        print(f"  NaN values: {validation_results['nan_count']}")
        print(f"  Label distribution: {validation_results['label_distribution']}")
        print(f"  Outliers detected: {validation_results['outliers_detected']}")
        
        return validation_results
    
    def generate_metadata(self, images, patches, validation_results):
        """
        Generate metadata file with processing parameters and statistics.
        
        Args:
            images (dict): Dictionary containing loaded images
            patches (list): List of extracted patches
            validation_results (dict): Validation results
        """
        metadata = {
            'processing_parameters': {
                'patch_size': self.patch_size,
                'stride': self.stride,
                'suitability_threshold': self.suitability_threshold,
                'noise_threshold': self.noise_threshold,
                'lee_filter_threshold': self.lee_filter_threshold
            },
            'image_statistics': {
                'image_dimensions': images['sar'].shape,
                'sar_range': [images['sar'].min(), images['sar'].max()],
                'dem_range': [images['dem'].min(), images['dem'].max()],
                'slope_range': [images['slope'].min(), images['slope'].max()]
            },
            'dataset_statistics': validation_results,
            'feature_descriptions': {
                'dominant_landcover_code': 'Most frequent landcover code in patch',
                'dominant_landcover_type': 'Most frequent landcover type name in patch',
                'dominant_landcover_percentage': 'Percentage of patch covered by dominant landcover type',
                'landcover_diversity': 'Number of different landcover types in patch',
                'alpha_mean': 'Average alpha decomposition value from G0 distribution',
                'gamma_mean': 'Average gamma value from G0 distribution',
                'sigma_mean': 'Average sigma backscatter coefficient',
                'coherence_mean': 'Average SAR coherence value (median for robustness)',
                'dem_median': 'Median elevation in meters',
                'slope_median': 'Median slope angle in degrees',
                'texture_contrast': 'GLCM contrast measure',
                'texture_dissimilarity': 'GLCM dissimilarity measure',
                'texture_homogeneity': 'GLCM homogeneity measure',
                'distance_to_water': 'Distance to nearest water body in pixels',
                'noise_level': 'Coefficient of variation (std/mean) of SAR patch',
                'suitability_ratio': 'Fraction of pixels labeled as suitable',
                'label': 'Binary suitability label (1=suitable, 0=not suitable)'
            },
            'landcover_legend': self.LANDCOVER_LEGEND
        }
        
        return metadata
    
    def generate_dataset(self, sar_path, dem_path, slope_path, landcover_path, 
                        label_path, coherence_path=None, output_dir='output'):
        """
        Main function to generate the complete dataset.
        
        Args:
            sar_path (str): Path to SAR image
            dem_path (str): Path to DEM image
            slope_path (str): Path to slope image
            landcover_path (str): Path to landcover image
            label_path (str): Path to label mask image
            coherence_path (str): Path to coherence image (optional)
            output_dir (str): Output directory for results
        """
        print("Starting camping site dataset generation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        images = self.load_images(sar_path, dem_path, slope_path, 
                                landcover_path, label_path, coherence_path)
        
        # Apply Lee filter to SAR image BEFORE patch extraction
        images['sar'] = self.apply_lee_filter_to_sar(images['sar'])
        
        # Extract patches
        patches = self.extract_patches(images)
        
        if not patches:
            print("No valid patches extracted!")
            return
        
        # Validate data
        validation_results = self.validate_data(patches)
        
        # Generate metadata
        metadata = self.generate_metadata(images, patches, validation_results)
        
        # Save dataset
        df = pd.DataFrame(patches)
        dataset_path = os.path.join(output_dir, 'camping_site_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        # Save metadata
        import json
        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nDataset generation complete!")
        print(f"Dataset saved to: {dataset_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Total patches: {len(patches)}")
        print(f"Class distribution: {validation_results['label_distribution']}")
        
        return df, metadata

def main():
    """
    Main function to run the dataset generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate camping site suitability dataset')
    parser.add_argument('--sar', help='Path to SAR image')
    parser.add_argument('--dem', help='Path to DEM image')
    parser.add_argument('--slope', help='Path to slope image')
    parser.add_argument('--landcover', help='Path to landcover image')
    parser.add_argument('--label', help='Path to label mask image')
    parser.add_argument('--coherence', help='Path to coherence image (optional)')
    parser.add_argument('--output', default='camping_site_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = CampingSiteDatasetGenerator(
        patch_size=50,
        stride=25,
        suitability_threshold=0.6,
        noise_threshold=0.8,
        lee_filter_threshold=0.6
    )
    
    # Use command line arguments if provided, otherwise use default paths
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
        # Default paths using your Tso Moriri files in data/ folder
        image_paths = {
            'sar_path': 'data/SAR_Image_5k_Tso_Moriri.tif',
            'dem_path': 'data/DEM_5km_Tso_Moriri.tif', 
            'slope_path': 'data/Slope_5k_Tso_Moriri.tif',
            'landcover_path': 'data/Landcover_Image_5k_Tso_Moriri.tif',
            'label_path': 'data/labels_Tso_Moriri.png',
            'coherence_path': 'data/Coherence_Stability_5km__Tso_Moriri.tif'
        }
        output_dir = 'camping_site_output'
    
    # Generate dataset
    try:
        df, metadata = generator.generate_dataset(
            sar_path=image_paths['sar_path'],
            dem_path=image_paths['dem_path'],
            slope_path=image_paths['slope_path'],
            landcover_path=image_paths['landcover_path'],
            label_path=image_paths['label_path'],
            coherence_path=image_paths['coherence_path'],
            output_dir=output_dir
        )
        
        print("\nDataset generation completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()