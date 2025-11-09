# Model Training Prompt for Cursor

Use this prompt when asking Cursor to create your model training script:

---

**Create a Python script to train a machine learning model for ASD (Autism Spectrum Disorder) prediction from eye-tracking data.**

## Dataset Format
The training CSV has the following columns in this exact order:
- `Tracking_F_1`, `Tracking_F_2`, `Tracking_F_3`, `Tracking_F_4`
- `Pupil_Diam_1` through `Pupil_Diam_12` (12 columns)
- `GazePoint_of_I_1` through `GazePoint_of_I_12` (12 columns)
- `Recording_1`, `Recording_2`, `Recording_3`
- `gaze_hori_1`, `gaze_hori_2`, `gaze_hori_3`, `gaze_hori_4`
- `gaze_vert_1`, `gaze_vert_2`, `gaze_vert_3`, `gaze_vert_4`
- `gaze_velo_1`, `gaze_velo_2`, `gaze_velo_3`, `gaze_velo_4`
- `blink_count_1`, `blink_count_2`, `blink_count_3`, `blink_count_4`
- `fix_count_1`, `fix_count_2`, `fix_count_3`, `fix_count_4`
- `sac_count_1`, `sac_count_2`, `sac_count_3`, `sac_count_4`
- `Source_File`, `level_2` (categorical - exclude from features)
- `trial_dur_1`, `trial_dur_2`
- `sampling_rate_1`
- `blink_rate_1`, `fixation_rate_1`, `saccade_rate_1`
- `fix_dur_avg_1`, `sac_amp_avg_1`, `sac_peak_vel_avg_1`
- `right_eye_c_1`, `right_eye_c_2`, `right_eye_c_3`
- `left_eye_c_1`, `left_eye_c_2`, `left_eye_c_3`
- `avg_eye_c_1`, `avg_eye_c_2`, `avg_eye_c_3`
- `pupil_diam_avg_1`, `gaze_hori_avg_1`, `gaze_vert_avg_1`
- `Participant` (exclude - not available in production)
- `Gender` (categorical - exclude, use `Gender_encoded` instead)
- `Age` (include as feature)
- `Class` (target variable - exclude from features)
- `CARS_Score_is_ASD` (exclude - not available in production)
- `Gender_encoded` (include as feature)

## Feature Selection Requirements

**EXCLUDE these columns from model features (they're metadata/identifiers, not features):**
- `Participant` - Participant ID (not available in production)
- `Source_File` - Source file identifier (categorical metadata)
- `level_2` - Categorical metadata
- `Gender` - Use `Gender_encoded` instead
- `Class` - This is the target variable (ASD/TD)
- `CARS_Score_is_ASD` - Clinical score not available in production

**INCLUDE all other numeric columns as features, in this exact order:**
1. `Tracking_F_1`, `Tracking_F_2`, `Tracking_F_3`, `Tracking_F_4`
2. `Pupil_Diam_1` through `Pupil_Diam_12` (all 12)
3. `GazePoint_of_I_1` through `GazePoint_of_I_12` (all 12)
4. `Recording_1`, `Recording_2`, `Recording_3`
5. `gaze_hori_1`, `gaze_hori_2`, `gaze_hori_3`, `gaze_hori_4`
6. `gaze_vert_1`, `gaze_vert_2`, `gaze_vert_3`, `gaze_vert_4`
7. `gaze_velo_1`, `gaze_velo_2`, `gaze_velo_3`, `gaze_velo_4`
8. `blink_count_1`, `blink_count_2`, `blink_count_3`, `blink_count_4`
9. `fix_count_1`, `fix_count_2`, `fix_count_3`, `fix_count_4`
10. `sac_count_1`, `sac_count_2`, `sac_count_3`, `sac_count_4`
11. `trial_dur_1`, `trial_dur_2`
12. `sampling_rate_1`
13. `blink_rate_1`, `fixation_rate_1`, `saccade_rate_1`
14. `fix_dur_avg_1`, `sac_amp_avg_1`, `sac_peak_vel_avg_1`
15. `right_eye_c_1`, `right_eye_c_2`, `right_eye_c_3`
16. `left_eye_c_1`, `left_eye_c_2`, `left_eye_c_3`
17. `avg_eye_c_1`, `avg_eye_c_2`, `avg_eye_c_3`
18. `pupil_diam_avg_1`, `gaze_hori_avg_1`, `gaze_vert_avg_1`
19. `Age`
20. `Gender_encoded`

**Total features: 75 numeric features**

## Target Variable
- Target column: `Class`
- Binary classification: `'ASD'` vs `'TD'` (Typical Development)
- Convert to binary: `is_ASD = 1` if `Class == 'ASD'`, else `0`

## Model Requirements
1. Use TensorFlow/Keras to build a neural network
2. Input shape: `(75,)` - 75 features in the exact order listed above
3. Output: Single neuron with sigmoid activation (binary classification)
4. Use StandardScaler to normalize features before training
5. Save the scaler parameters (mean, scale, feature_names) to `public/scaler_params.json`
6. Export model to TensorFlow.js format in `public/model/` directory
7. The feature order in the scaler must match the exact order above

## Feature Order Validation
The script must:
1. Read the CSV and verify all expected columns exist
2. Extract features in the EXACT order listed above (75 features total)
3. Exclude the metadata columns (`Participant`, `Source_File`, `level_2`, `Gender`, `Class`, `CARS_Score_is_ASD`)
4. Ensure `Gender_encoded` and `Age` are included in the feature set
5. Print the feature names in order to verify correctness
6. Save the feature order to the scaler params JSON so the frontend can match it

## Output Files
- `public/scaler_params.json` - Contains `scalerParams` (mean, scale arrays) and `feature_names` array in exact order
- `public/model/model.json` - TensorFlow.js model file
- `public/model/*.bin` - Model weight files

## Important Notes
- The frontend will generate features in the exact same order as defined above
- The model must accept features in this exact order
- Feature names must match exactly (case-sensitive)
- Do NOT include any columns that won't be available in production (Participant, CARS_Score_is_ASD, etc.)

