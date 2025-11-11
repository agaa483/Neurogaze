# Setup Guide - Neurogaze Eye Tracking App

This guide will walk you through getting the Neurogaze application fully operational.

## Prerequisites

1. **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
2. **Python 3.8+** (for model training) - [Download](https://www.python.org/)
3. **A webcam** (for eye tracking)
4. **Training dataset** (CSV file with eye-tracking data for ASD/TD classification)

## Step 1: Install Frontend Dependencies

```bash
cd Neurogaze
npm install
```

This installs:
- React and Vite
- MediaPipe FaceMesh libraries
- Other dependencies

## Step 2: Train the Machine Learning Model

### 2.1 Install Python Dependencies

Create a Python virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install tensorflow pandas numpy scikit-learn
```

### 2.2 Create Training Script

You need to create a Python script (`train_model.py`) that:
- Reads your training CSV file
- Extracts 75 features in the exact order specified in `MODEL_TRAINING.md`
- Trains a TensorFlow/Keras neural network
- Saves the model in TensorFlow.js format to `public/model/`
- Saves scaler parameters to `public/scaler_params.json`

**Note:** The training script doesn't exist yet - you'll need to create it based on `MODEL_TRAINING.md` specifications.

### 2.3 Run Training

```bash
python train_model.py --data your_training_data.csv
```

This should create:
- `public/model/model.json` and weight files
- `public/scaler_params.json`

## Step 3: Add TensorFlow.js to Frontend

The frontend currently doesn't have TensorFlow.js integrated. You'll need to:

1. Install TensorFlow.js:
```bash
npm install @tensorflow/tfjs
```

2. Add model loading and prediction code to `App.jsx`

## Step 4: Run the Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173` (or the port Vite assigns).

## Step 5: Test the Application

1. **Grant camera access** when prompted
2. **Enter participant info**: Age (2-18) and Gender
3. **Run calibration**: Click "Start Calibration" and follow the 5 points
4. **Start assessment**: Click "Start 30s Capture" - images will display automatically
5. **Download CSV**: After 30 seconds, download the CSV file
6. **View prediction**: (Once model integration is complete) See ASD/TD prediction

## Current Status

✅ **Working:**
- Eye tracking with MediaPipe
- Calibration system
- 30-second data capture
- CSV export with 75 features
- Image display during assessment

❌ **Missing:**
- Python training script (`train_model.py`)
- Trained model files (`public/model/` directory)
- Scaler parameters (`public/scaler_params.json`)
- TensorFlow.js integration in frontend
- Prediction display after assessment

## Next Steps

1. **Create the training script** - Follow `MODEL_TRAINING.md` specifications
2. **Train the model** - Use your training dataset
3. **Integrate TensorFlow.js** - Add model loading and prediction to the React app
4. **Test end-to-end** - Verify predictions work correctly

## Troubleshooting

### Camera not working
- Ensure you're using HTTPS or localhost (required for camera access)
- Check browser permissions
- Try a different browser (Chrome/Edge recommended)

### Model not loading
- Check that `public/model/model.json` exists
- Verify `public/scaler_params.json` exists
- Check browser console for errors

### Training script issues
- Verify your CSV has all required columns
- Check feature order matches exactly
- Ensure Python dependencies are installed

