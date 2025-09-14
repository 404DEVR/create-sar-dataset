# Patch Filtering and Label ID Explanation

## 🎯 **What Was Added**

### 1. **Label ID Field**
- **New Column**: `label_id` - Shows the most common original label value in each patch
- **Purpose**: Track what the original label mask contained for each patch
- **Values**: 60 (202 patches), 80 (178 patches), 100 (38 patches)

### 2. **Clean Terminal Output**
- ✅ Removed verbose water detection debug messages
- ✅ Removed individual patch processing messages
- ✅ Added clean filtering statistics summary

### 3. **Patch Filtering Statistics**
- Shows exactly why patches were discarded
- Tracks filtering reasons and percentages

## 🔍 **How Patches Are Filtered/Discarded**

### **Filtering Criteria (in order):**

1. **High Noise Filter** (`noise_threshold = 1.2`)
   - **What**: Patches with SAR noise level ≥ 1.2
   - **Why**: High noise indicates poor data quality
   - **Result**: 0 patches discarded (0.0%)

2. **Invalid Terrain Filter**
   - **What**: Patches with NaN values in DEM or slope
   - **Why**: Invalid terrain data makes features unreliable
   - **Result**: 0 patches discarded (0.0%)

3. **Valid Patches**
   - **What**: Patches that pass all filters
   - **Result**: 418 patches kept (100.0%)

## 📊 **Label ID Explanation**

### **Label ID Values:**
- **60**: Most common label in 202 patches (48.3%)
- **80**: Most common label in 178 patches (42.6%) 
- **100**: Most common label in 38 patches (9.1%)

### **Label ID vs Final Label:**
- **Label ID**: Original label mask value (what was in the input image)
- **Final Label**: Binary suitability label (0/1) after processing
- **Label Source**: How the final label was determined:
  - `original`: From original label mask
  - `terrain`: From terrain-based criteria (slope, DEM, landcover)

## 🎯 **Current Results Summary**

### **Dataset Statistics:**
- **Total Patches**: 418
- **Success Rate**: 100% (no patches discarded)
- **Label Distribution**: 258 unsuitable (0), 160 suitable (1)
- **Label Sources**: 258 original, 160 terrain-based

### **Key Features:**
- **Alpha Range**: -0.930 to -0.500 (highly variable)
- **Water Distance**: 1.54 to 657.63 pixels (387 unique values)
- **Water Detection**: Correctly identifies code 80 as water bodies
- **Label ID**: Tracks original label mask values for each patch

## 🔧 **Technical Details**

### **Patch Extraction Process:**
1. Extract 50x50 pixel patches with 25-pixel stride
2. Calculate SAR features (alpha, gamma, sigma, noise)
3. Calculate terrain features (DEM, slope)
4. Calculate texture features (GLCM)
5. Calculate spatial features (distance to water)
6. Generate labels using multiple approaches
7. Apply quality filters
8. Save valid patches to CSV

### **Label Generation Strategy:**
1. **Primary**: Use original label mask (suitability_ratio)
2. **Fallback**: Use terrain criteria (slope < 20°, DEM < 5000m, suitable landcover)
3. **Alternative**: Use relaxed criteria (slope < 15°, DEM < 4000m)
4. **Final**: Binary label (1 if suitability_ratio > 0.3, else 0)

The system now provides complete transparency about patch filtering and label generation! 🎯
