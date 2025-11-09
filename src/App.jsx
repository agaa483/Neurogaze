import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  FaceMesh,
  FACEMESH_LEFT_EYE,
  FACEMESH_RIGHT_EYE,
} from '@mediapipe/face_mesh'
import { Camera } from '@mediapipe/camera_utils'
import { drawConnectors } from '@mediapipe/drawing_utils'
import abstractImage from './assets/images/abstract.jpg'
import cartoonNatureImage from './assets/images/cartoonnature.jpg'
import portraitImage from './assets/images/portrait.jpg'

const leftIrisIndices = [468, 469, 470, 471]
const rightIrisIndices = [473, 474, 475, 476]

const leftEyeTopIndices = [159, 160]
const leftEyeBottomIndices = [145, 144]
const leftEyeInnerCorner = 133
const leftEyeOuterCorner = 33

const rightEyeTopIndices = [386, 387]
const rightEyeBottomIndices = [374, 380]
const rightEyeInnerCorner = 362
const rightEyeOuterCorner = 263

const statusCopy = {
  idle: 'Waiting for camera',
  requesting: 'Requesting camera access…',
  ready: 'Tracking eyes',
  error: 'Camera error',
  unsupported: 'Camera not supported',
}

const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

const classifyMovement = (velocity, openness, blinkThreshold, fixationThreshold) => {
  if (openness < blinkThreshold) {
    return 'Blink'
  }
  if (velocity < fixationThreshold) {
    return 'Fixation'
  }
  return 'Saccade'
}

const AVERAGE_EYE_WIDTH_MM = 30
const BLINK_THRESHOLD = 0.18
const FIXATION_VELOCITY_THRESHOLD = 0.015

const calibrationPoints = [
  { x: 0.5, y: 0.5, label: 'center' },
  { x: 0.2, y: 0.2, label: 'top-left' },
  { x: 0.8, y: 0.2, label: 'top-right' },
  { x: 0.2, y: 0.8, label: 'bottom-left' },
  { x: 0.8, y: 0.8, label: 'bottom-right' },
]

const FRAMES_PER_CAL_POINT = 45

const defaultCalibrationModel = () => ({
  scaleX: 1,
  offsetX: 0,
  scaleY: 1,
  offsetY: 0,
})

const mean = (values) =>
  values.reduce((acc, value) => acc + value, 0) / (values.length || 1)

const computeAxisMapping = (measuredValues, targetValues) => {
  const meanMeasured = mean(measuredValues)
  const meanTarget = mean(targetValues)
  const variance =
    measuredValues.reduce(
      (acc, value) => acc + (value - meanMeasured) * (value - meanMeasured),
      0
    ) || 0

  if (variance < 1e-6) {
    return { scale: 1, offset: meanTarget - meanMeasured }
  }

  const covariance = measuredValues.reduce(
    (acc, value, index) =>
      acc + (value - meanMeasured) * (targetValues[index] - meanTarget),
    0
  )

  const scale = covariance / variance
  const offset = meanTarget - scale * meanMeasured
  return { scale, offset }
}

const computeLinearMapping = (pairs) => {
  if (!pairs.length) {
    return defaultCalibrationModel()
  }
  const measuredX = pairs.map((pair) => pair.measured.x)
  const targetX = pairs.map((pair) => pair.target.x)
  const measuredY = pairs.map((pair) => pair.measured.y)
  const targetY = pairs.map((pair) => pair.target.y)

  const mapX = computeAxisMapping(measuredX, targetX)
  const mapY = computeAxisMapping(measuredY, targetY)

  return {
    scaleX: mapX.scale,
    offsetX: mapX.offset,
    scaleY: mapY.scale,
    offsetY: mapY.offset,
  }
}

const ASSESSMENT_DURATION_MS = 30_000
const IMAGE_DISPLAY_DURATION_MS = 10_000 // 10 seconds per image
const ASSESSMENT_IMAGES = [
  { src: abstractImage, name: 'Abstract Art' },
  { src: cartoonNatureImage, name: 'Cartoon Nature' },
  { src: portraitImage, name: 'Portrait' },
]

// Aggregated feature headers matching training data format
const AGGREGATED_CSV_HEADERS = [
  'Tracking_F_1', 'Tracking_F_2', 'Tracking_F_3', 'Tracking_F_4',
  'Pupil_Diam_1', 'Pupil_Diam_2', 'Pupil_Diam_3', 'Pupil_Diam_4', 'Pupil_Diam_5', 'Pupil_Diam_6',
  'Pupil_Diam_7', 'Pupil_Diam_8', 'Pupil_Diam_9', 'Pupil_Diam_10', 'Pupil_Diam_11', 'Pupil_Diam_12',
  'GazePoint_of_I_1', 'GazePoint_of_I_2', 'GazePoint_of_I_3', 'GazePoint_of_I_4', 'GazePoint_of_I_5', 'GazePoint_of_I_6',
  'GazePoint_of_I_7', 'GazePoint_of_I_8', 'GazePoint_of_I_9', 'GazePoint_of_I_10', 'GazePoint_of_I_11', 'GazePoint_of_I_12',
  'Recording_1', 'Recording_2', 'Recording_3',
  'gaze_hori_1', 'gaze_hori_2', 'gaze_hori_3', 'gaze_hori_4',
  'gaze_vert_1', 'gaze_vert_2', 'gaze_vert_3', 'gaze_vert_4',
  'gaze_velo_1', 'gaze_velo_2', 'gaze_velo_3', 'gaze_velo_4',
  'blink_count_1', 'blink_count_2', 'blink_count_3', 'blink_count_4',
  'fix_count_1', 'fix_count_2', 'fix_count_3', 'fix_count_4',
  'sac_count_1', 'sac_count_2', 'sac_count_3', 'sac_count_4',
  'Source_File', 'level_2',
  'trial_dur_1', 'trial_dur_2',
  'sampling_rate_1',
  'blink_rate_1', 'fixation_rate_1', 'saccade_rate_1',
  'fix_dur_avg_1', 'sac_amp_avg_1', 'sac_peak_vel_avg_1',
  'right_eye_c_1', 'right_eye_c_2', 'right_eye_c_3',
  'left_eye_c_1', 'left_eye_c_2', 'left_eye_c_3',
  'avg_eye_c_1', 'avg_eye_c_2', 'avg_eye_c_3',
  'pupil_diam_avg_1', 'gaze_hori_avg_1', 'gaze_vert_avg_1',
  'Participant', 'Gender', 'Age', 'Class', 'CARS_Score_is_ASD', 'Gender_encoded',
]

// Helper function to compute statistics
const computeStats = (values) => {
  if (!values || values.length === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, q1: 0, q3: 0, median: 0 }
  }
  const validValues = values.filter(v => v != null && !Number.isNaN(v) && Number.isFinite(v))
  if (validValues.length === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, q1: 0, q3: 0, median: 0 }
  }
  const sorted = [...validValues].sort((a, b) => a - b)
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
  const variance = sorted.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / sorted.length
  const std = Math.sqrt(variance)
  const min = sorted[0]
  const max = sorted[sorted.length - 1]
  const q1 = sorted[Math.floor(sorted.length * 0.25)]
  const q3 = sorted[Math.floor(sorted.length * 0.75)]
  const median = sorted[Math.floor(sorted.length * 0.5)]
  return { mean, std, min, max, q1, q3, median }
}

// Feature engineering function to match training data format
const computeAggregatedFeatures = (samples, age, gender) => {
  if (!samples || samples.length === 0) {
    return null
  }

  // Extract arrays for each metric
  const trackingRatios = samples.map(s => s.trackingRatio).filter(v => v != null)
  const pupilRight = samples.map(s => s.pupilDiameterRightMm).filter(v => v != null && v > 0)
  const pupilLeft = samples.map(s => s.pupilDiameterLeftMm).filter(v => v != null && v > 0)
  const porRightX = samples.map(s => s.pointOfRegardRightX).filter(v => v != null)
  const porRightY = samples.map(s => s.pointOfRegardRightY).filter(v => v != null)
  const porLeftX = samples.map(s => s.pointOfRegardLeftX).filter(v => v != null)
  const porLeftY = samples.map(s => s.pointOfRegardLeftY).filter(v => v != null)
  const recordingTimes = samples.map(s => s.recordingTimeMs).filter(v => v != null)

  // Compute gaze velocities and movements
  const gazeVelocitiesRight = []
  const gazeVelocitiesLeft = []
  const gazeHorizontalRight = []
  const gazeVerticalRight = []
  const gazeHorizontalLeft = []
  const gazeVerticalLeft = []
  const fixDurations = []
  const saccadeAmplitudes = []
  const saccadePeakVelocities = []

  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1]
    const curr = samples[i]
    const dt = (curr.recordingTimeMs - prev.recordingTimeMs) / 1000

    if (dt > 0) {
      // Right eye velocity
      const dxRight = curr.pointOfRegardRightX - prev.pointOfRegardRightX
      const dyRight = curr.pointOfRegardRightY - prev.pointOfRegardRightY
      const velocityRight = Math.hypot(dxRight, dyRight) / dt
      gazeVelocitiesRight.push(velocityRight)
      gazeHorizontalRight.push(dxRight / dt)
      gazeVerticalRight.push(dyRight / dt)

      // Left eye velocity
      const dxLeft = curr.pointOfRegardLeftX - prev.pointOfRegardLeftX
      const dyLeft = curr.pointOfRegardLeftY - prev.pointOfRegardLeftY
      const velocityLeft = Math.hypot(dxLeft, dyLeft) / dt
      gazeVelocitiesLeft.push(velocityLeft)
      gazeHorizontalLeft.push(dxLeft / dt)
      gazeVerticalLeft.push(dyLeft / dt)

      // Saccade amplitude (distance moved)
      if (curr.categoryRight === 'Saccade' || curr.categoryLeft === 'Saccade') {
        const amplitude = Math.hypot(dxRight + dxLeft, dyRight + dyLeft) / 2
        saccadeAmplitudes.push(amplitude)
        saccadePeakVelocities.push(Math.max(velocityRight, velocityLeft))
      }

      // Fixation duration (time spent in fixation)
      if (curr.categoryRight === 'Fixation' && prev.categoryRight === 'Fixation') {
        fixDurations.push(dt)
      }
    }
  }

  // Count eye movement categories (split into 4 time segments)
  const segmentSize = Math.ceil(samples.length / 4)
  const blinkCounts = [0, 0, 0, 0]
  const fixCounts = [0, 0, 0, 0]
  const sacCounts = [0, 0, 0, 0]

  samples.forEach((sample, idx) => {
    const segment = Math.min(3, Math.floor(idx / segmentSize))
    if (sample.categoryRight === 'Blink' || sample.categoryLeft === 'Blink') blinkCounts[segment]++
    if (sample.categoryRight === 'Fixation' || sample.categoryLeft === 'Fixation') fixCounts[segment]++
    if (sample.categoryRight === 'Saccade' || sample.categoryLeft === 'Saccade') sacCounts[segment]++
  })

  // Compute statistics
  const trackingStats = computeStats(trackingRatios)
  const pupilRightStats = computeStats(pupilRight)
  const pupilLeftStats = computeStats(pupilLeft)
  const porRightXStats = computeStats(porRightX)
  const porRightYStats = computeStats(porRightY)
  const porLeftXStats = computeStats(porLeftX)
  const porLeftYStats = computeStats(porLeftY)
  const recordingTimeStats = {
    count: recordingTimes.length,
    min: recordingTimes.length > 0 ? Math.min(...recordingTimes) : 0,
    max: recordingTimes.length > 0 ? Math.max(...recordingTimes) : 0,
  }

  const gazeHorizontalRightStats = computeStats(gazeHorizontalRight)
  const gazeVerticalRightStats = computeStats(gazeVerticalRight)
  const gazeVelocityRightStats = computeStats(gazeVelocitiesRight)
  const gazeHorizontalLeftStats = computeStats(gazeHorizontalLeft)
  const gazeVerticalLeftStats = computeStats(gazeVerticalLeft)
  const gazeVelocityLeftStats = computeStats(gazeVelocitiesLeft)

  // Combined gaze stats
  const gazeHorizontalMean = (gazeHorizontalRightStats.mean + gazeHorizontalLeftStats.mean) / 2
  const gazeHorizontalStd = Math.sqrt((Math.pow(gazeHorizontalRightStats.std, 2) + Math.pow(gazeHorizontalLeftStats.std, 2)) / 2)
  const gazeVerticalMean = (gazeVerticalRightStats.mean + gazeVerticalLeftStats.mean) / 2
  const gazeVerticalStd = Math.sqrt((Math.pow(gazeVerticalRightStats.std, 2) + Math.pow(gazeVerticalLeftStats.std, 2)) / 2)

  // Trial duration
  const trialDurationMs = recordingTimeStats.max - recordingTimeStats.min
  const trialDurationSec = trialDurationMs > 0 ? trialDurationMs / 1000 : 1
  const samplingRate = trialDurationSec > 0 ? samples.length / trialDurationSec : 0

  // Rates per second
  const blinkRate = trialDurationSec > 0 ? blinkCounts.reduce((a, b) => a + b, 0) / trialDurationSec : 0
  const fixationRate = trialDurationSec > 0 ? fixCounts.reduce((a, b) => a + b, 0) / trialDurationSec : 0
  const saccadeRate = trialDurationSec > 0 ? sacCounts.reduce((a, b) => a + b, 0) / trialDurationSec : 0

  // Average fixation duration, saccade amplitude, saccade peak velocity
  const fixDurAvg = fixDurations.length > 0 ? fixDurations.reduce((a, b) => a + b, 0) / fixDurations.length : 0
  const sacAmpAvg = saccadeAmplitudes.length > 0 ? saccadeAmplitudes.reduce((a, b) => a + b, 0) / saccadeAmplitudes.length : 0
  const sacPeakVelAvg = saccadePeakVelocities.length > 0 ? saccadePeakVelocities.reduce((a, b) => a + b, 0) / saccadePeakVelocities.length : 0

  // Eye coordinates (mean, std, range)
  const rightEyeC = {
    mean: porRightXStats.mean,
    std: porRightXStats.std,
    range: porRightXStats.max - porRightXStats.min,
  }
  const leftEyeC = {
    mean: porLeftXStats.mean,
    std: porLeftXStats.std,
    range: porLeftXStats.max - porLeftXStats.min,
  }
  const avgEyeC = {
    mean: (porRightXStats.mean + porLeftXStats.mean) / 2,
    std: (porRightXStats.std + porLeftXStats.std) / 2,
    range: ((porRightXStats.max - porRightXStats.min) + (porLeftXStats.max - porLeftXStats.min)) / 2,
  }

  // Gender encoding (M=1, F=0, based on typical encoding)
  const genderEncoded = gender === 'M' ? 1 : (gender === 'F' ? 0 : 0.5)

  // Build feature array in exact order matching AGGREGATED_CSV_HEADERS
  return {
    'Tracking_F_1': trackingStats.mean,
    'Tracking_F_2': trackingStats.std,
    'Tracking_F_3': trackingStats.min,
    'Tracking_F_4': trackingStats.max,
    'Pupil_Diam_1': pupilRightStats.mean,
    'Pupil_Diam_2': pupilRightStats.std,
    'Pupil_Diam_3': pupilRightStats.min,
    'Pupil_Diam_4': pupilRightStats.max,
    'Pupil_Diam_5': pupilRightStats.q1,
    'Pupil_Diam_6': pupilRightStats.q3,
    'Pupil_Diam_7': pupilLeftStats.mean,
    'Pupil_Diam_8': pupilLeftStats.std,
    'Pupil_Diam_9': pupilLeftStats.min,
    'Pupil_Diam_10': pupilLeftStats.max,
    'Pupil_Diam_11': pupilLeftStats.q1,
    'Pupil_Diam_12': pupilLeftStats.q3,
    'GazePoint_of_I_1': porRightXStats.mean,
    'GazePoint_of_I_2': porRightXStats.std,
    'GazePoint_of_I_3': porRightXStats.min,
    'GazePoint_of_I_4': porRightXStats.max,
    'GazePoint_of_I_5': porRightYStats.mean,
    'GazePoint_of_I_6': porRightYStats.std,
    'GazePoint_of_I_7': porRightYStats.min,
    'GazePoint_of_I_8': porRightYStats.max,
    'GazePoint_of_I_9': porLeftXStats.mean,
    'GazePoint_of_I_10': porLeftXStats.std,
    'GazePoint_of_I_11': porLeftXStats.min,
    'GazePoint_of_I_12': porLeftXStats.max,
    'Recording_1': recordingTimeStats.count,
    'Recording_2': recordingTimeStats.min,
    'Recording_3': recordingTimeStats.max,
    'gaze_hori_1': gazeHorizontalRightStats.mean,
    'gaze_hori_2': gazeHorizontalRightStats.std,
    'gaze_hori_3': gazeHorizontalLeftStats.mean,
    'gaze_hori_4': gazeHorizontalLeftStats.std,
    'gaze_vert_1': gazeVerticalRightStats.mean,
    'gaze_vert_2': gazeVerticalRightStats.std,
    'gaze_vert_3': gazeVerticalLeftStats.mean,
    'gaze_vert_4': gazeVerticalLeftStats.std,
    'gaze_velo_1': gazeVelocityRightStats.mean,
    'gaze_velo_2': gazeVelocityRightStats.max,
    'gaze_velo_3': gazeVelocityLeftStats.mean,
    'gaze_velo_4': gazeVelocityLeftStats.max,
    'blink_count_1': blinkCounts[0],
    'blink_count_2': blinkCounts[1],
    'blink_count_3': blinkCounts[2],
    'blink_count_4': blinkCounts[3],
    'fix_count_1': fixCounts[0],
    'fix_count_2': fixCounts[1],
    'fix_count_3': fixCounts[2],
    'fix_count_4': fixCounts[3],
    'sac_count_1': sacCounts[0],
    'sac_count_2': sacCounts[1],
    'sac_count_3': sacCounts[2],
    'sac_count_4': sacCounts[3],
    'Source_File': 'web-app',
    'level_2': 'GazePoint_of_I',
    'trial_dur_1': trialDurationMs,
    'trial_dur_2': trialDurationSec,
    'sampling_rate_1': samplingRate,
    'blink_rate_1': blinkRate,
    'fixation_rate_1': fixationRate,
    'saccade_rate_1': saccadeRate,
    'fix_dur_avg_1': fixDurAvg,
    'sac_amp_avg_1': sacAmpAvg,
    'sac_peak_vel_avg_1': sacPeakVelAvg,
    'right_eye_c_1': rightEyeC.mean,
    'right_eye_c_2': rightEyeC.std,
    'right_eye_c_3': rightEyeC.range,
    'left_eye_c_1': leftEyeC.mean,
    'left_eye_c_2': leftEyeC.std,
    'left_eye_c_3': leftEyeC.range,
    'avg_eye_c_1': avgEyeC.mean,
    'avg_eye_c_2': avgEyeC.std,
    'avg_eye_c_3': avgEyeC.range,
    'pupil_diam_avg_1': (pupilRightStats.mean + pupilLeftStats.mean) / 2,
    'gaze_hori_avg_1': gazeHorizontalMean,
    'gaze_vert_avg_1': gazeVerticalMean,
    'Participant': 0, // Placeholder - not used in model
    'Gender': gender || 'Unknown',
    'Age': parseFloat(age) || 0,
    'Class': 'Unknown', // Will be predicted by model
    'CARS_Score_is_ASD': 0, // Placeholder - not used in model
    'Gender_encoded': genderEncoded,
  }
}

const formatCsvCell = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return ''
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return ''
    }
    if (Number.isInteger(value)) {
      return value.toString()
    }
    return Number(value.toFixed(3)).toString()
  }
  const str = String(value).replace(/"/g, '""')
  return `"${str}"`
}

const triggerCsvDownload = (url, prefix = 'eye-tracking') => {
  if (typeof window === 'undefined') {
    return
  }
  const link = document.createElement('a')
  link.href = url
  link.download = `${prefix}-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

const revokeObjectUrl = (url) => {
  if (url) {
    URL.revokeObjectURL(url)
  }
}

const applyCalibration = (model, point) => {
  if (!point || !model) {
    return point ?? null
  }
  return {
    x: clamp(model.scaleX * point.x + model.offsetX, 0, 1),
    y: clamp(model.scaleY * point.y + model.offsetY, 0, 1),
  }
}

const averagePoint = (samples, key) => {
  const valid = samples
    .map((sample) => sample[key])
    .filter((samplePoint) => !!samplePoint)
  if (!valid.length) {
    return null
  }
  const accumulator = valid.reduce(
    (acc, point) => ({
      x: acc.x + point.x,
      y: acc.y + point.y,
    }),
    { x: 0, y: 0 }
  )
  return {
    x: accumulator.x / valid.length,
    y: accumulator.y / valid.length,
  }
}

const getAverageLandmark = (landmarks, indices) => {
  const points = indices
    .map((index) => landmarks[index])
    .filter((point) => point)

  if (!points.length) {
    return null
  }

  const sum = points.reduce(
    (acc, point) => ({
      x: acc.x + point.x,
      y: acc.y + point.y,
      z: acc.z + (point.z ?? 0),
    }),
    { x: 0, y: 0, z: 0 }
  )

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length,
  }
}

const getDistance = (a, b) => {
  if (!a || !b) {
    return 0
  }
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const metricsInternalRef = useRef({
    blinkState: 'open',
    blinkTimestamps: [],
    lastUpdate: 0,
    blinkCount: 0,
    prevLeftCenter: null,
    prevRightCenter: null,
    prevTimestamp: null,
    samplesTotal: 0,
    samplesValid: 0,
    startTime: Date.now(),
    leftIrisRadiusPx: 0,
    rightIrisRadiusPx: 0,
  })
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState({
    recordingTimeMs: 0,
    categoryRight: 'Unknown',
    categoryLeft: 'Unknown',
    pupilDiameterRightMm: 0,
    pupilDiameterLeftMm: 0,
    pointOfRegardRightX: 0,
    pointOfRegardRightY: 0,
    pointOfRegardLeftX: 0,
    pointOfRegardLeftY: 0,
    trackingRatio: 0,
  })
  const [calibration, setCalibration] = useState({
    status: 'idle',
    currentIndex: 0,
    totalPoints: calibrationPoints.length,
    targetLabel: calibrationPoints[0].label,
  })
  const calibrationRef = useRef({
    status: 'idle',
    pointIndex: 0,
    samplesForPoint: [],
    recorded: [],
    models: {
      left: defaultCalibrationModel(),
      right: defaultCalibrationModel(),
    },
  })
  const isSupported = useMemo(
    () => !!navigator?.mediaDevices?.getUserMedia,
    []
  )
  const assessmentRef = useRef({
    status: 'idle',
    samples: [],
    startTimestamp: 0,
    downloadUrl: '',
  })
  const [assessment, setAssessment] = useState({
    status: 'idle',
    timeLeftMs: ASSESSMENT_DURATION_MS,
    samplesCaptured: 0,
    downloadUrl: '',
  })
  const [userInfo, setUserInfo] = useState({
    age: '',
    gender: '',
  })
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  const resetCalibration = useCallback(() => {
    calibrationRef.current = {
      status: 'idle',
      pointIndex: 0,
      samplesForPoint: [],
      recorded: [],
      models: {
        left: defaultCalibrationModel(),
        right: defaultCalibrationModel(),
      },
    }
    setCalibration({
      status: 'idle',
      currentIndex: 0,
      totalPoints: calibrationPoints.length,
      targetLabel: calibrationPoints[0].label,
    })
  }, [])

  const startCalibration = useCallback(() => {
    if (!isSupported) {
      setError('Camera access is required for calibration.')
      return
    }
    calibrationRef.current.status = 'running'
    calibrationRef.current.pointIndex = 0
    calibrationRef.current.samplesForPoint = []
    calibrationRef.current.recorded = []
    calibrationRef.current.models = {
      left: defaultCalibrationModel(),
      right: defaultCalibrationModel(),
    }
    setCalibration({
      status: 'running',
      currentIndex: 0,
      totalPoints: calibrationPoints.length,
      targetLabel: calibrationPoints[0].label,
    })
  }, [isSupported, setCalibration])

  const finalizeAssessment = useCallback(() => {
    if (assessmentRef.current.status !== 'running') {
      return
    }

    assessmentRef.current.status = 'complete'
    const samples = assessmentRef.current.samples.slice()

    // Compute aggregated features matching training data format
    const aggregatedFeatures = computeAggregatedFeatures(samples, userInfo.age, userInfo.gender)

    if (!aggregatedFeatures) {
      setError('Unable to compute features from samples.')
      return
    }

    // Create CSV with aggregated features
    const csvLines = [AGGREGATED_CSV_HEADERS.join(',')]
    const row = AGGREGATED_CSV_HEADERS.map(header => {
      const value = aggregatedFeatures[header]
      return formatCsvCell(value)
    }).join(',')
    csvLines.push(row)

    const blob = new Blob([csvLines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    revokeObjectUrl(assessmentRef.current.downloadUrl)
    assessmentRef.current.downloadUrl = url

    setAssessment({
      status: 'complete',
      timeLeftMs: 0,
      samplesCaptured: samples.length,
      downloadUrl: url,
    })
  }, [userInfo.age, userInfo.gender])

  const resetAssessment = useCallback(() => {
    revokeObjectUrl(assessmentRef.current.downloadUrl)
    setAssessment({
      status: 'idle',
      timeLeftMs: ASSESSMENT_DURATION_MS,
      samplesCaptured: 0,
      downloadUrl: '',
    })
    assessmentRef.current = {
      status: 'idle',
      samples: [],
      startTimestamp: 0,
      downloadUrl: '',
    }
    setCurrentImageIndex(0)
  }, [])

  const startAssessment = useCallback(() => {
    if (!isSupported) {
      setError('Camera access is required to capture data.')
      return
    }
    if (!userInfo.age || !userInfo.gender) {
      setError('Please enter age and gender before starting the assessment.')
      return
    }
    const ageNum = parseFloat(userInfo.age)
    if (isNaN(ageNum) || ageNum < 2 || ageNum > 18) {
      setError('Age must be between 2 and 18.')
      return
    }
    if (assessmentRef.current.status === 'running') {
      return
    }
    revokeObjectUrl(assessment.downloadUrl)
    revokeObjectUrl(assessmentRef.current.downloadUrl)

    assessmentRef.current = {
      status: 'running',
      samples: [],
      startTimestamp: performance.now(),
      downloadUrl: '',
    }

    setAssessment({
      status: 'running',
      timeLeftMs: ASSESSMENT_DURATION_MS,
      samplesCaptured: 0,
      downloadUrl: '',
    })
    setError('')
  }, [assessment.downloadUrl, isSupported, userInfo.age, userInfo.gender, setAssessment, setError])

  const handleCsvDownload = useCallback(() => {
    if (assessment.downloadUrl) {
      triggerCsvDownload(assessment.downloadUrl, 'eye-tracking')
    }
  }, [assessment.downloadUrl])

  useEffect(() => {
    if (assessment.status !== 'running') {
      return undefined
    }

    assessmentRef.current.status = 'running'
    if (!assessmentRef.current.startTimestamp) {
      assessmentRef.current.startTimestamp = performance.now()
    }

    // Reset to first image when assessment starts
    setCurrentImageIndex(0)

    let rafId = 0
    const tick = () => {
      if (assessmentRef.current.status !== 'running') {
        return
      }
      const elapsed = performance.now() - assessmentRef.current.startTimestamp
      const remaining = Math.max(0, ASSESSMENT_DURATION_MS - elapsed)
      
      // Calculate which image should be shown (10 seconds per image)
      const imageIndex = Math.min(
        ASSESSMENT_IMAGES.length - 1,
        Math.floor(elapsed / IMAGE_DISPLAY_DURATION_MS)
      )
      setCurrentImageIndex(imageIndex)
      
      setAssessment((prev) =>
        prev.status === 'running' ? { ...prev, timeLeftMs: remaining } : prev
      )
      
      if (remaining <= 0) {
        finalizeAssessment()
        return
      }
      
      rafId = requestAnimationFrame(tick)
    }

    rafId = requestAnimationFrame(tick)
    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId)
      }
    }
  }, [assessment.status, finalizeAssessment])

  useEffect(() => {
    return () => {
      revokeObjectUrl(assessment.downloadUrl)
    }
  }, [assessment.downloadUrl])

  useEffect(() => {
    if (!isSupported) {
      setStatus('unsupported')
      setError('This browser does not expose the required camera APIs.')
      return
    }

    metricsInternalRef.current.startTime = Date.now()
    let camera = null
    let isActive = true
    const videoElement = videoRef.current
    const canvasElement = canvasRef.current
    const canvasCtx = canvasElement?.getContext('2d')

    if (!videoElement || !canvasElement || !canvasCtx) {
      setError('Unable to access the video or canvas elements.')
      setStatus('error')
      return
    }

    let faceMesh
    try {
      faceMesh = new FaceMesh({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      })

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
    } catch (faceMeshError) {
      console.error('Unable to initialize MediaPipe FaceMesh', faceMeshError)
      setError(
        faceMeshError instanceof Error
          ? faceMeshError.message
          : 'Unable to initialize the eye tracking model.'
      )
      setStatus('error')
      return
    }

    const getIrisGeometry = (landmarks, irisIndices, width, height) => {
      const points = irisIndices
        .map((index) => landmarks[index])
        .filter(Boolean)

      if (points.length !== irisIndices.length) {
        return null
      }

      const center = points.reduce(
        (acc, point) => ({
          x: acc.x + point.x,
          y: acc.y + point.y,
        }),
        { x: 0, y: 0 }
      )
      center.x /= points.length
      center.y /= points.length

      const radius =
        points.reduce((acc, point) => {
          const dx = (point.x - center.x) * width
          const dy = (point.y - center.y) * height
          return acc + Math.hypot(dx, dy)
        }, 0) / points.length || 0

      return {
        center,
        radius,
        pixel: { x: center.x * width, y: center.y * height },
      }
    }

    const drawIris = (geometry, color) => {
      if (!geometry) {
        return
      }
      canvasCtx.fillStyle = color
      canvasCtx.beginPath()
      canvasCtx.arc(
        geometry.pixel.x,
        geometry.pixel.y,
        Math.max(2, geometry.radius * 0.6),
        0,
        2 * Math.PI
      )
      canvasCtx.fill()
    }

    const onResults = (results) => {
      if (!isActive) {
        return
      }

      const { image, multiFaceLandmarks } = results
      if (!image) {
        return
      }

      canvasElement.width = image.width
      canvasElement.height = image.height

      canvasCtx.save()
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)
      
      // Only draw video frame if not in assessment (when images are showing)
      const isInAssessment = assessmentRef.current.status === 'running'
      if (!isInAssessment) {
        canvasCtx.scale(-1, 1)
        canvasCtx.translate(-canvasElement.width, 0)
        canvasCtx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height)
        canvasCtx.restore()
        canvasCtx.save()
        // Re-apply transform for eye tracking visualization
        canvasCtx.scale(-1, 1)
        canvasCtx.translate(-canvasElement.width, 0)
      }

      metricsInternalRef.current.samplesTotal += 1

      if (multiFaceLandmarks && multiFaceLandmarks.length > 0) {
        metricsInternalRef.current.samplesValid += 1
        const landmarks = multiFaceLandmarks[0]
        
        // Apply transform for eye tracking visualization
        if (isInAssessment) {
          canvasCtx.scale(-1, 1)
          canvasCtx.translate(-canvasElement.width, 0)
        }
        
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {
          color: '#14ffec',
          lineWidth: 1.5,
        })
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {
          color: '#14ffec',
          lineWidth: 1.5,
        })
        const leftGeometry = getIrisGeometry(
          landmarks,
          leftIrisIndices,
          canvasElement.width,
          canvasElement.height
        )
        const rightGeometry = getIrisGeometry(
          landmarks,
          rightIrisIndices,
          canvasElement.width,
          canvasElement.height
        )
        drawIris(leftGeometry, '#ff4ecd')
        drawIris(rightGeometry, '#ff4ecd')

        const leftIrisCenter = getAverageLandmark(landmarks, leftIrisIndices)
        const rightIrisCenter = getAverageLandmark(landmarks, rightIrisIndices)
        const leftTop = getAverageLandmark(landmarks, leftEyeTopIndices)
        const leftBottom = getAverageLandmark(landmarks, leftEyeBottomIndices)
        const rightTop = getAverageLandmark(landmarks, rightEyeTopIndices)
        const rightBottom = getAverageLandmark(landmarks, rightEyeBottomIndices)
        const leftInner = landmarks[leftEyeInnerCorner]
        const leftOuter = landmarks[leftEyeOuterCorner]
        const rightInner = landmarks[rightEyeInnerCorner]
        const rightOuter = landmarks[rightEyeOuterCorner]

        const calibrationCtx = calibrationRef.current
        if (
          calibrationCtx.status === 'running' &&
          leftIrisCenter &&
          rightIrisCenter
        ) {
          calibrationCtx.samplesForPoint.push({
            left: leftIrisCenter,
            right: rightIrisCenter,
          })
          if (calibrationCtx.samplesForPoint.length >= FRAMES_PER_CAL_POINT) {
            const averageLeft = averagePoint(
              calibrationCtx.samplesForPoint,
              'left'
            )
            const averageRight = averagePoint(
              calibrationCtx.samplesForPoint,
              'right'
            )
            calibrationCtx.recorded.push({
              target: calibrationPoints[calibrationCtx.pointIndex],
              left: averageLeft,
              right: averageRight,
            })
            calibrationCtx.samplesForPoint = []
            calibrationCtx.pointIndex += 1

            if (calibrationCtx.pointIndex >= calibrationPoints.length) {
              const leftPairs = calibrationCtx.recorded
                .filter((entry) => entry.left)
                .map((entry) => ({
                  measured: entry.left,
                  target: entry.target,
                }))
              const rightPairs = calibrationCtx.recorded
                .filter((entry) => entry.right)
                .map((entry) => ({
                  measured: entry.right,
                  target: entry.target,
                }))

              calibrationCtx.models = {
                left: leftPairs.length
                  ? computeLinearMapping(leftPairs)
                  : defaultCalibrationModel(),
                right: rightPairs.length
                  ? computeLinearMapping(rightPairs)
                  : defaultCalibrationModel(),
              }
              calibrationCtx.status = 'complete'
              setCalibration({
                status: 'complete',
                currentIndex: calibrationPoints.length,
                totalPoints: calibrationPoints.length,
                targetLabel: null,
              })
              if (typeof window !== 'undefined') {
                window.dispatchEvent(
                  new CustomEvent('eye-calibration-complete', {
                    detail: {
                      leftModel: calibrationCtx.models.left,
                      rightModel: calibrationCtx.models.right,
                    },
                  })
                )
              }
            } else {
              setCalibration({
                status: 'running',
                currentIndex: calibrationCtx.pointIndex,
                totalPoints: calibrationPoints.length,
                targetLabel:
                  calibrationPoints[calibrationCtx.pointIndex].label,
              })
            }
          }
        }

        const leftHorizontal = getDistance(leftInner, leftOuter)
        const rightHorizontal = getDistance(rightInner, rightOuter)
        const leftVertical = getDistance(leftTop, leftBottom)
        const rightVertical = getDistance(rightTop, rightBottom)

        const leftOpenness =
          leftHorizontal > 0 ? clamp(leftVertical / leftHorizontal, 0, 1) : 0
        const rightOpenness =
          rightHorizontal > 0 ? clamp(rightVertical / rightHorizontal, 0, 1) : 0

        const averageOpenness = (leftOpenness + rightOpenness) / 2

        const internal = metricsInternalRef.current
        const now = performance.now()

        const blinkOpenThreshold = 0.24

        if (averageOpenness < BLINK_THRESHOLD && internal.blinkState === 'open') {
          internal.blinkState = 'closed'
          internal.blinkCount += 1
          internal.blinkTimestamps.push(Date.now())
        } else if (
          averageOpenness > blinkOpenThreshold &&
          internal.blinkState === 'closed'
        ) {
          internal.blinkState = 'open'
        }

        internal.blinkTimestamps = internal.blinkTimestamps.filter(
          (timestamp) => Date.now() - timestamp <= 60000
        )

        if (now - internal.lastUpdate > 100) {
          internal.lastUpdate = now
          const currentTimestamp = Date.now()
          const elapsedMs = currentTimestamp - internal.startTime
          const deltaMs = internal.prevTimestamp
            ? Math.max(1, currentTimestamp - internal.prevTimestamp)
            : 1

          const leftVelocity =
            internal.prevLeftCenter && leftIrisCenter
              ? (Math.hypot(
                  leftIrisCenter.x - internal.prevLeftCenter.x,
                  leftIrisCenter.y - internal.prevLeftCenter.y
                ) /
                  deltaMs) *
                1000
              : 0
          const rightVelocity =
            internal.prevRightCenter && rightIrisCenter
              ? (Math.hypot(
                  rightIrisCenter.x - internal.prevRightCenter.x,
                  rightIrisCenter.y - internal.prevRightCenter.y
                ) /
                  deltaMs) *
                1000
              : 0

          internal.prevLeftCenter = leftIrisCenter
          internal.prevRightCenter = rightIrisCenter
          internal.prevTimestamp = currentTimestamp

          const leftEyeWidthPx = leftHorizontal * canvasElement.width
          const rightEyeWidthPx = rightHorizontal * canvasElement.width
          const leftScale =
            leftEyeWidthPx > 0 ? AVERAGE_EYE_WIDTH_MM / leftEyeWidthPx : 0
          const rightScale =
            rightEyeWidthPx > 0 ? AVERAGE_EYE_WIDTH_MM / rightEyeWidthPx : 0

          const leftPupilDiameterMm =
            leftGeometry && leftScale
              ? Number((leftGeometry.radius * 2 * leftScale).toFixed(2))
              : 0
          const rightPupilDiameterMm =
            rightGeometry && rightScale
              ? Number((rightGeometry.radius * 2 * rightScale).toFixed(2))
              : 0

          const correctedLeft = applyCalibration(
            calibrationRef.current.models.left,
            leftIrisCenter
          )
          const correctedRight = applyCalibration(
            calibrationRef.current.models.right,
            rightIrisCenter
          )

          const pointOfRegardLeft = correctedLeft
            ? {
                x: Number((correctedLeft.x * image.width).toFixed(0)),
                y: Number((correctedLeft.y * image.height).toFixed(0)),
              }
            : { x: 0, y: 0 }

          const pointOfRegardRight = correctedRight
            ? {
                x: Number((correctedRight.x * image.width).toFixed(0)),
                y: Number((correctedRight.y * image.height).toFixed(0)),
              }
            : { x: 0, y: 0 }

          const categoryLeft = classifyMovement(
            leftVelocity,
            leftOpenness,
            BLINK_THRESHOLD,
            FIXATION_VELOCITY_THRESHOLD
          )
          const categoryRight = classifyMovement(
            rightVelocity,
            rightOpenness,
            BLINK_THRESHOLD,
            FIXATION_VELOCITY_THRESHOLD
          )

          const trackingRatio =
            metricsInternalRef.current.samplesTotal > 0
              ? Number(
                  (
                    (metricsInternalRef.current.samplesValid /
                      metricsInternalRef.current.samplesTotal) *
                    100
                  ).toFixed(2)
                )
              : 0

          const nextMetrics = {
            recordingTimeMs: elapsedMs,
            categoryRight,
            categoryLeft,
            pupilDiameterRightMm: rightPupilDiameterMm,
            pupilDiameterLeftMm: leftPupilDiameterMm,
            pointOfRegardRightX: pointOfRegardRight.x,
            pointOfRegardRightY: pointOfRegardRight.y,
            pointOfRegardLeftX: pointOfRegardLeft.x,
            pointOfRegardLeftY: pointOfRegardLeft.y,
            trackingRatio,
          }

          if (assessmentRef.current.status === 'running') {
            const relativeRecordingMs = Math.min(
              ASSESSMENT_DURATION_MS,
              performance.now() - assessmentRef.current.startTimestamp
            )

            assessmentRef.current.samples.push({
              recordingTimeMs: Number(relativeRecordingMs.toFixed(2)),
              timestampIso: new Date().toISOString(),
              categoryRight,
              categoryLeft,
              pupilDiameterRightMm: Number(
                nextMetrics.pupilDiameterRightMm.toFixed(2)
              ),
              pupilDiameterLeftMm: Number(
                nextMetrics.pupilDiameterLeftMm.toFixed(2)
              ),
              pointOfRegardRightX: nextMetrics.pointOfRegardRightX,
              pointOfRegardRightY: nextMetrics.pointOfRegardRightY,
              pointOfRegardLeftX: nextMetrics.pointOfRegardLeftX,
              pointOfRegardLeftY: nextMetrics.pointOfRegardLeftY,
              trackingRatio: Number(nextMetrics.trackingRatio.toFixed(2)),
            })

            setAssessment((prev) =>
              prev.status === 'running'
                ? {
                    ...prev,
                    samplesCaptured: assessmentRef.current.samples.length,
                  }
                : prev
            )
          }

          setMetrics(nextMetrics)

          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('eye-tracking-data', { detail: nextMetrics })
            )
          }
        }
      }

      canvasCtx.restore()
    }

    faceMesh.onResults(onResults)

    const startCamera = async () => {
      setStatus('requesting')
      try {
        camera = new Camera(videoElement, {
          onFrame: async () => {
            if (!isActive) {
              return
            }
            await faceMesh.send({ image: videoElement })
          },
          width: 640,
          height: 480,
        })
        await camera.start()
        setStatus('ready')
      } catch (err) {
        if (!isActive) {
          return
        }
        setError(
          err instanceof Error
            ? err.message
            : 'Unable to connect to the camera.'
        )
        setStatus('error')
      }
    }

    startCamera()

    return () => {
      isActive = false
      if (faceMesh) {
        faceMesh.close()
      }
      if (camera) {
        camera.stop()
      }
      const stream = videoElement.srcObject
      if (stream) {
        const tracks = stream.getTracks ? stream.getTracks() : []
        tracks.forEach((track) => track.stop())
      }
      videoElement.srcObject = null
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)
    }
  }, [isSupported])

  const statusLabel =
    statusCopy[status] ?? (status === 'error' ? 'Camera error' : 'Status')
  const activeCalibrationPoint =
    calibration.status === 'running' && calibration.currentIndex < calibrationPoints.length
      ? calibrationPoints[calibration.currentIndex]
      : null
  const calibrationProgress =
    calibration.status === 'running'
      ? calibration.currentIndex / calibration.totalPoints
      : calibration.status === 'complete'
        ? 1
        : 0
  const assessmentSecondsRemaining = Math.max(
    0,
    Math.ceil(assessment.timeLeftMs / 1000)
  )
  const isCalibrationReady = calibration.status === 'complete'
  const hasUserInfo = userInfo.age && userInfo.gender
  const ageValid = hasUserInfo && !isNaN(parseFloat(userInfo.age)) && parseFloat(userInfo.age) >= 2 && parseFloat(userInfo.age) <= 18
  const canStartAssessment = assessment.status !== 'running' && isCalibrationReady && hasUserInfo && ageValid

  return (
    <div className="app">
      <header className="header">
        <h1>Eye Tracker</h1>
        <p>
          Grant camera access to visualize real-time eye tracking powered by
          MediaPipe FaceMesh.
        </p>
      </header>

      <section className="user-info-section">
        <div className="user-info-header">
          <h2>Participant Information</h2>
          <p>Please provide age and gender for accurate analysis</p>
        </div>
        <div className="user-info-inputs">
          <div className="input-group">
            <label htmlFor="age">Age</label>
            <input
              id="age"
              type="number"
              min="2"
              max="18"
              value={userInfo.age}
              onChange={(e) => {
                const val = e.target.value
                if (val === '' || (parseFloat(val) >= 2 && parseFloat(val) <= 18)) {
                  setUserInfo({ ...userInfo, age: val })
                }
              }}
              placeholder="Enter age (2-18)"
              disabled={assessment.status === 'running'}
            />
          </div>
          <div className="input-group">
            <label htmlFor="gender">Gender</label>
            <select
              id="gender"
              value={userInfo.gender}
              onChange={(e) => setUserInfo({ ...userInfo, gender: e.target.value })}
              disabled={assessment.status === 'running'}
            >
              <option value="">Select gender</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
        </div>
        {(!userInfo.age || !userInfo.gender) && assessment.status === 'idle' && (
          <p className="user-info-hint">
            Please enter age (2-18) and gender before starting the assessment.
          </p>
        )}
      </section>

      <section className="calibration-controls">
        <div className="calibration-copy">
          <h2>Calibration</h2>
          <p>
            Improve gaze accuracy by following the highlighted targets. Allow
            roughly 1–2 seconds per point.
          </p>
        </div>
        <div className="calibration-actions">
          <button
            type="button"
            className="control-btn primary"
            onClick={startCalibration}
            disabled={calibration.status === 'running'}
          >
            {calibration.status === 'running' ? 'Calibrating…' : 'Start Calibration'}
          </button>
          {calibration.status === 'complete' && (
            <button
              type="button"
              className="control-btn secondary"
              onClick={resetCalibration}
            >
              Recalibrate
            </button>
          )}
        </div>
        {calibration.status === 'running' && (
          <>
            <div className="calibration-progress">
              <span style={{ width: `${calibrationProgress * 100}%` }} />
            </div>
            <p className="calibration-message">
              Focus on the {calibration.targetLabel?.replace('-', ' ')} target (
              {Math.min(calibration.currentIndex + 1, calibration.totalPoints)} /{' '}
              {calibration.totalPoints})
            </p>
          </>
        )}
        {calibration.status === 'complete' && (
          <p className="calibration-message success">
            Calibration complete. Metrics now use corrected gaze coordinates.
          </p>
        )}
      </section>

      <section className="assessment-runner">
        <div className="assessment-header">
          <h2>30-Second Data Capture</h2>
          <p>
            Collects calibrated gaze samples for 30 seconds and exports them to
            CSV automatically.
          </p>
        </div>
        <div className="assessment-timer">
          <span className="timer-value">
            {assessmentSecondsRemaining.toString().padStart(2, '0')}
          </span>
          <span className="timer-label">seconds left</span>
        </div>
        <div className="assessment-details">
          <span>Status: {assessment.status}</span>
          <span>Samples: {assessment.samplesCaptured}</span>
        </div>
        <div className="assessment-buttons">
          <button
            type="button"
            className="control-btn primary"
            onClick={startAssessment}
            disabled={!canStartAssessment}
          >
            {assessment.status === 'running'
              ? 'Capturing…'
              : 'Start 30s Capture'}
          </button>
          <button
            type="button"
            className="control-btn secondary"
            onClick={resetAssessment}
            disabled={
              assessment.status === 'idle' &&
              assessment.samplesCaptured === 0 &&
              !assessment.downloadUrl
            }
          >
            {assessment.status === 'running' ? 'Cancel Capture' : 'Reset'}
          </button>
          {assessment.status === 'complete' && assessment.downloadUrl && (
            <button
              type="button"
              className="control-btn secondary"
              onClick={handleCsvDownload}
            >
              Download CSV
            </button>
          )}
        </div>
        {!isCalibrationReady && (
          <p className="assessment-hint">
            Complete calibration before starting the capture run.
          </p>
        )}
        {(!hasUserInfo || !ageValid) && assessment.status === 'idle' && (
          <p className="assessment-hint">
            Please enter valid age (2-18) and gender in the Participant Information section above.
          </p>
        )}
        {assessment.status === 'complete' && (
          <p className="assessment-hint success">
            Capture finished. Ready to download CSV with {assessment.samplesCaptured}{' '}
            samples.
          </p>
        )}
      </section>

      <div className="viewer">
        {assessment.status === 'running' && ASSESSMENT_IMAGES[currentImageIndex] && (
          <div className="assessment-image-container">
            <img
              src={ASSESSMENT_IMAGES[currentImageIndex].src}
              alt={ASSESSMENT_IMAGES[currentImageIndex].name}
              className="assessment-image"
            />
          </div>
        )}
        <video
          ref={videoRef}
          className={`video ${assessment.status === 'running' ? 'video-hidden' : ''}`}
          playsInline
          muted
          autoPlay
          aria-hidden
        />
        <canvas ref={canvasRef} className="overlay" />
        <span className={`status-badge status-${status}`}>{statusLabel}</span>
        {activeCalibrationPoint && (
          <div
            className="calibration-target"
            style={{
              left: `${activeCalibrationPoint.x * 100}%`,
              top: `${activeCalibrationPoint.y * 100}%`,
            }}
          />
        )}
        {assessment.status === 'running' && (
          <div className="image-navigation">
            <div className="image-counter">
              Image {currentImageIndex + 1} of {ASSESSMENT_IMAGES.length}
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          <strong>Heads up:</strong> {error}
        </div>
      )}

      <section className="metrics">
        <h2>Live Biomarkers</h2>
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-label">Recording time</span>
            <span className="metric-value">
              {(metrics.recordingTimeMs / 1000).toFixed(2)}s
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Right eye state</span>
            <span className="metric-value">{metrics.categoryRight}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Left eye state</span>
            <span className="metric-value">{metrics.categoryLeft}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Right pupil diameter</span>
            <span className="metric-value">
              {metrics.pupilDiameterRightMm.toFixed(2)} mm
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Left pupil diameter</span>
            <span className="metric-value">
              {metrics.pupilDiameterLeftMm.toFixed(2)} mm
            </span>
          </div>
          <div className="metric-card metric-card--wide">
            <span className="metric-label">Right gaze (X, Y)</span>
            <span className="metric-value">
              {metrics.pointOfRegardRightX}, {metrics.pointOfRegardRightY}
            </span>
            <span className="metric-subtext">pixels</span>
          </div>
          <div className="metric-card metric-card--wide">
            <span className="metric-label">Left gaze (X, Y)</span>
            <span className="metric-value">
              {metrics.pointOfRegardLeftX}, {metrics.pointOfRegardLeftY}
            </span>
            <span className="metric-subtext">pixels</span>
          </div>
          <div className="metric-card metric-card--wide">
            <span className="metric-label">Tracking ratio</span>
            <span className="metric-value">{metrics.trackingRatio.toFixed(2)}%</span>
          </div>
        </div>
        <p className="metric-footnote">
          Metrics stream with each MediaPipe frame. Tracking ratio shows the
          share of valid samples; run calibration to align gaze coordinates
          before exporting data.
        </p>
      </section>

      <footer className="footer">
        <p>
          Tip: Make sure you have good lighting and keep your face within the
          frame for best results.
        </p>
      </footer>
    </div>
  )
}

export default App
