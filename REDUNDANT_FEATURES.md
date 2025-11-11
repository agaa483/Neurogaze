# Redundant Features to Remove

Based on analysis of the web app code and feature structure, here are features that are **redundant** and should be removed from BOTH the training dataset and web app output.

## ❌ Features to Remove (Redundant Averages)

These features are **perfect linear combinations** of other features. The model can learn the same information from the source features.

### 1. Average Eye Coordinates (3 features)
- `avg_eye_c_1` = (right_eye_c_1 + left_eye_c_1) / 2
- `avg_eye_c_2` = (right_eye_c_2 + left_eye_c_2) / 2  
- `avg_eye_c_3` = (right_eye_c_3 + left_eye_c_3) / 2

**Reason**: Simple average of right and left eye coordinates. The model can learn from individual eyes and compute averages if needed.

**Keep instead**: `right_eye_c_1-3` and `left_eye_c_1-3`

### 2. Average Pupil Diameter (1 feature)
- `pupil_diam_avg_1` = (Pupil_Diam_1 + Pupil_Diam_7) / 2

**Reason**: Simple average of right and left pupil diameter means. Already have individual eye statistics.

**Keep instead**: `Pupil_Diam_1` (right mean) and `Pupil_Diam_7` (left mean)

### 3. Average Gaze Velocities (2 features)
- `gaze_hori_avg_1` = (gaze_hori_1 + gaze_hori_3) / 2
- `gaze_vert_avg_1` = (gaze_vert_1 + gaze_vert_3) / 2

**Reason**: Simple averages of right and left eye gaze velocities. Already have per-eye statistics.

**Keep instead**: `gaze_hori_1-4` and `gaze_vert_1-4` (per-eye statistics)

## ⚠️ Potentially Redundant (Consider Removing)

### 4. Trial Duration in Seconds (1 feature)
- `trial_dur_2` = `trial_dur_1` / 1000

**Reason**: This is just a unit conversion. However, having both might help with different scales. **Recommendation**: Keep both for now, but could remove `trial_dur_2` if you want to reduce features.

## Summary

**Total features to remove: 6**
- `avg_eye_c_1`
- `avg_eye_c_2`
- `avg_eye_c_3`
- `pupil_diam_avg_1`
- `gaze_hori_avg_1`
- `gaze_vert_avg_1`

**New feature count: 69 features** (down from 75)

## Why Remove These?

1. **Perfect linear combinations**: The model can learn the same patterns from the source features
2. **Reduces overfitting**: Fewer features = simpler model = less overfitting risk
3. **Faster training**: Fewer features = faster training and inference
4. **No information loss**: All information is preserved in the source features

## Implementation

After removing these, you'll need to:
1. Update `clean_dataset.py` to exclude these features
2. Update `App.jsx` to not generate these features
3. Update `MODEL_TRAINING.md` to reflect 69 features instead of 75
4. Retrain the model with 69 features

