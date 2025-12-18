# Proctoring Detection Threshold Calibration Guide

## Overview
All detection thresholds are now centralized at the top of `proctoring_module.py` for easy tuning and calibration.

## Threshold Configuration Reference

### 🔹 Eye Blink Detection

**Constant:** `BLINK_THRESHOLD = 3.6`

**What it measures:** Eye Aspect Ratio (EAR) - ratio of horizontal to vertical eye distance

**How it works:**
- Lower value = More sensitive (detects partial blinks)
- Higher value = Less sensitive (requires more eye closure)

**Typical range:** 3.0 - 4.5

**Calibration tips:**
- If too many false positives (detecting blinks when eyes are open): **Increase** value
- If missing actual blinks: **Decrease** value
- Test with different lighting conditions

---

### 🔹 Gaze Detection

**Constants:**
- `GAZE_RATIO_THRESHOLD = 1.2`
- `GAZE_THRESHOLD_VALUE = 50`

**What it measures:**
- `GAZE_RATIO_THRESHOLD`: Ratio of white pixels on one side vs other side of eye
- `GAZE_THRESHOLD_VALUE`: Binary threshold for eye segmentation (pupil detection)

**How it works:**
- `GAZE_RATIO_THRESHOLD`:
  - Lower value = More sensitive to small eye movements
  - Higher value = Requires more extreme eye movement
- `GAZE_THRESHOLD_VALUE`:
  - Lower value = More sensitive to darker pixels (pupil/iris)
  - Higher value = Less sensitive

**Typical ranges:**
- `GAZE_RATIO_THRESHOLD`: 1.0 - 1.5
- `GAZE_THRESHOLD_VALUE`: 40 - 70

**Calibration tips:**
- If detecting "Left"/"Right" when looking center: **Increase** `GAZE_RATIO_THRESHOLD`
- If not detecting side glances: **Decrease** `GAZE_RATIO_THRESHOLD`
- If pupil not detected properly: Adjust `GAZE_THRESHOLD_VALUE` based on lighting

---

### 🔹 Mouth Detection

**Constant:** `MOUTH_OPEN_THRESHOLD = 23`

**What it measures:** Distance in pixels between outer top and bottom lip

**How it works:**
- Lower value = Detects smaller mouth openings
- Higher value = Only detects wide mouth openings

**Typical range:** 15 - 30 pixels (depends on camera distance and resolution)

**Calibration tips:**
- If detecting mouth open when it's closed: **Increase** value
- If not detecting talking/yawning: **Decrease** value
- Distance depends on:
  - Camera resolution
  - Distance from camera
  - Face size in frame

---

### 🔹 Head Pose Detection

**Constants:**
- `HEAD_VERTICAL_ANGLE_THRESHOLD = 30`
- `HEAD_HORIZONTAL_OFFSET_THRESHOLD = 10`

**What it measures:**
- `HEAD_VERTICAL_ANGLE_THRESHOLD`: Angle in degrees for up/down head movement
- `HEAD_HORIZONTAL_OFFSET_THRESHOLD`: Pixel offset for left/right head movement

**How it works:**
- `HEAD_VERTICAL_ANGLE_THRESHOLD`:
  - Lower value = More sensitive to slight tilts
  - Higher value = Requires more extreme head tilt
- `HEAD_HORIZONTAL_OFFSET_THRESHOLD`:
  - Lower value = More sensitive to slight turns
  - Higher value = Requires more extreme head turn

**Typical ranges:**
- `HEAD_VERTICAL_ANGLE_THRESHOLD`: 20 - 40 degrees
- `HEAD_HORIZONTAL_OFFSET_THRESHOLD`: 5 - 15 pixels

**Calibration tips:**
- If detecting "Head Down"/"Head Up" too often: **Increase** vertical threshold
- If detecting "Head Left"/"Head Right" too often: **Increase** horizontal threshold
- If not detecting head movements: **Decrease** respective thresholds

---

### 🔹 Audio Detection (Voice)

**Constants:**
- `AUDIO_AMPLITUDE_MIN = 1000`
- `AUDIO_AMPLITUDE_MAX = 20000`
- `AUDIO_ZCR_MIN = 0.05`
- `AUDIO_ZCR_MAX = 0.35`
- `AUDIO_AMPLITUDE_VARIATION_MIN = 500`

**What it measures:**
- **Amplitude range:** Sound volume (16-bit audio: -32768 to +32767)
- **ZCR (Zero-Crossing Rate):** Frequency characteristic of audio signal
- **Amplitude variation:** Standard deviation to distinguish speech from constant noise

**How it works:**
- `AUDIO_AMPLITUDE_MIN`: Below this = silence/too quiet for speech
- `AUDIO_AMPLITUDE_MAX`: Above this = loud noise/music, not normal speech
- `AUDIO_ZCR_MIN/MAX`: Human voice typically has ZCR between 0.1-0.3
- `AUDIO_AMPLITUDE_VARIATION_MIN`: Speech has amplitude variation, static noise doesn't

**Typical ranges:**
- `AUDIO_AMPLITUDE_MIN`: 500 - 2000
- `AUDIO_AMPLITUDE_MAX`: 15000 - 25000
- `AUDIO_ZCR_MIN`: 0.03 - 0.1
- `AUDIO_ZCR_MAX`: 0.25 - 0.4
- `AUDIO_AMPLITUDE_VARIATION_MIN`: 300 - 800

**Calibration tips:**
- If detecting voice when room is silent: **Increase** `AUDIO_AMPLITUDE_MIN`
- If not detecting whispering: **Decrease** `AUDIO_AMPLITUDE_MIN`
- If detecting background music as voice: Adjust `AUDIO_ZCR_MIN/MAX` range
- If detecting constant fan/AC noise as voice: **Increase** `AUDIO_AMPLITUDE_VARIATION_MIN`

---

## Quick Calibration Process

### Step 1: Baseline Testing
1. Test with normal exam conditions (lighting, distance, environment)
2. Record false positives and false negatives
3. Note which thresholds are causing issues

### Step 2: Incremental Adjustment
1. Adjust one threshold at a time
2. Test change with multiple users
3. Document impact of change
4. Repeat until optimal

### Step 3: Environment-Specific Tuning
Consider adjusting for:
- **Camera quality/resolution** → Affects pixel-based thresholds (mouth, head horizontal)
- **Lighting conditions** → Affects gaze threshold and eye detection
- **Microphone sensitivity** → Affects audio amplitude thresholds
- **Background noise** → Affects audio ZCR and variation thresholds

### Step 4: Edge Case Testing
Test with:
- ✅ Different ethnicities (eye shapes)
- ✅ Glasses/contacts wearers
- ✅ Different hairstyles
- ✅ Poor lighting conditions
- ✅ Noisy environments
- ✅ Different camera angles

---

## Configuration Examples

### More Sensitive Detection (Stricter Proctoring)
```python
BLINK_THRESHOLD = 3.0                    # Detects more blinks
GAZE_RATIO_THRESHOLD = 1.0               # Detects smaller eye movements
MOUTH_OPEN_THRESHOLD = 18                # Detects smaller mouth openings
HEAD_VERTICAL_ANGLE_THRESHOLD = 20       # Detects smaller head tilts
HEAD_HORIZONTAL_OFFSET_THRESHOLD = 5     # Detects smaller head turns
AUDIO_AMPLITUDE_MIN = 800                # Detects quieter sounds
```

### Less Sensitive Detection (More Lenient)
```python
BLINK_THRESHOLD = 4.2                    # Requires more eye closure
GAZE_RATIO_THRESHOLD = 1.5               # Requires more extreme eye movement
MOUTH_OPEN_THRESHOLD = 28                # Requires wider mouth opening
HEAD_VERTICAL_ANGLE_THRESHOLD = 40       # Requires larger head tilts
HEAD_HORIZONTAL_OFFSET_THRESHOLD = 15    # Requires more head turn
AUDIO_AMPLITUDE_MIN = 1500               # Ignores quieter sounds
```

---

## Benefits of Centralized Thresholds

✅ **Easy Tuning:** Change all thresholds in one location  
✅ **Better Documentation:** Each threshold has explanation and typical range  
✅ **Consistent Behavior:** No magic numbers scattered throughout code  
✅ **Environment Adaptation:** Quick adjustment for different setups  
✅ **Testing:** Easy to create test configurations  
✅ **Maintenance:** Clear understanding of detection sensitivity  

---

## Monitoring and Optimization

### Recommended Metrics to Track
1. **False Positive Rate:** % of events flagged incorrectly
2. **False Negative Rate:** % of real violations missed
3. **Event Distribution:** Which events are most/least common
4. **User Feedback:** Student complaints about sensitivity

### Iterative Improvement
1. Deploy with default (current) values
2. Collect data on event frequency
3. Review flagged events for accuracy
4. Adjust thresholds based on patterns
5. Re-deploy and monitor

---

## Support

For questions about threshold calibration, refer to:
- **Code:** `server/proctoring_module.py` (lines 7-78)
- **Testing:** Test each detection function individually
- **Documentation:** This file and inline comments in code
