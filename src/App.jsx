import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  FaceMesh,
  FACEMESH_LEFT_EYE,
  FACEMESH_RIGHT_EYE,
} from '@mediapipe/face_mesh'
import { Camera } from '@mediapipe/camera_utils'
import { drawConnectors } from '@mediapipe/drawing_utils'

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
const CSV_HEADERS = [
  'recording_time_ms',
  'sample_timestamp_iso',
  'category_right',
  'category_left',
  'pupil_diameter_right_mm',
  'pupil_diameter_left_mm',
  'point_of_regard_right_x',
  'point_of_regard_right_y',
  'point_of_regard_left_x',
  'point_of_regard_left_y',
  'tracking_ratio_percent',
]

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

    const csvLines = [CSV_HEADERS.join(',')]
    samples.forEach((sample) => {
      csvLines.push(
        [
          sample.recordingTimeMs,
          sample.timestampIso,
          sample.categoryRight,
          sample.categoryLeft,
          sample.pupilDiameterRightMm,
          sample.pupilDiameterLeftMm,
          sample.pointOfRegardRightX,
          sample.pointOfRegardRightY,
          sample.pointOfRegardLeftX,
          sample.pointOfRegardLeftY,
          sample.trackingRatio,
        ]
          .map(formatCsvCell)
          .join(',')
      )
    })

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

    if (typeof window !== 'undefined') {
      requestAnimationFrame(() => triggerCsvDownload(url))
    }
  }, [])

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
  }, [])

  const startAssessment = useCallback(() => {
    if (!isSupported) {
      setError('Camera access is required to capture data.')
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
  }, [assessment.downloadUrl, isSupported, setAssessment, setError])

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

    let rafId = 0
    const tick = () => {
      if (assessmentRef.current.status !== 'running') {
        return
      }
      const elapsed = performance.now() - assessmentRef.current.startTimestamp
      const remaining = Math.max(0, ASSESSMENT_DURATION_MS - elapsed)
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
      canvasCtx.scale(-1, 1)
      canvasCtx.translate(-canvasElement.width, 0)
      canvasCtx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height)

      metricsInternalRef.current.samplesTotal += 1

      if (multiFaceLandmarks && multiFaceLandmarks.length > 0) {
        metricsInternalRef.current.samplesValid += 1
        const landmarks = multiFaceLandmarks[0]
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
  const canStartAssessment = assessment.status !== 'running' && isCalibrationReady

  return (
    <div className="app">
      <header className="header">
        <h1>Eye Tracker</h1>
        <p>
          Grant camera access to visualize real-time eye tracking powered by
          MediaPipe FaceMesh.
        </p>
      </header>

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
        {assessment.status === 'complete' && (
          <p className="assessment-hint success">
            Capture finished. CSV downloaded with {assessment.samplesCaptured}{' '}
            samples.
          </p>
        )}
      </section>

      <div className="viewer">
        <video
          ref={videoRef}
          className="video"
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
