# iOS App Deployment Guide - Personal Use

## Current Status: Ready to Deploy

The iOS app is **fully implemented**. All code is written and verified:
- Camera pipeline with AVCaptureSession
- Grid detection using Vision framework
- Perspective correction with CoreImage
- Cell extraction (81 cells at 28x28)
- CoreML v2 digit classifier
- C solver compiled into the app
- SwiftUI interface with solution overlay

---

## Step 1: Open the Project

```bash
open ios/SudokuVision.xcodeproj
```

## Step 2: Configure Code Signing

1. Select the **SudokuVision** target in the project navigator (left sidebar)
2. Go to **Signing & Capabilities** tab
3. Check **"Automatically manage signing"**
4. Select your **Team** (your Apple ID)
   - If not listed: Xcode → Settings → Accounts → Add your Apple ID

## Step 3: Connect iPhone and Build

1. Connect your iPhone via USB
2. Trust the computer on your iPhone if prompted
3. Select your iPhone as the run destination (dropdown at top of Xcode)
4. Press **⌘R** (or Product → Run)

## Step 4: Trust Developer Certificate on iPhone

After first install:

1. On iPhone: **Settings → General → VPN & Device Management**
2. Find your developer certificate (your Apple ID email)
3. Tap **Trust**
4. Launch the app

---

## Verification

1. **Launch app** - Should open to camera view
2. **Point at a sudoku puzzle** - Blue corner markers when grid detected
3. **Hold steady** - After 3 stable frames, shows solved digits in blue

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Untrusted Developer" | Settings → General → VPN & Device Management → Trust |
| "Could not launch app" | Ensure device is unlocked |
| Camera permission denied | Settings → SudokuVision → Enable Camera |
| Grid not detected | Good lighting, full puzzle visible, minimal glare |
| Digits misrecognized | Works best on clean printed puzzles |

---

## Notes

- **Free Apple ID** allows 7-day provisioning profiles (sufficient for personal use)
- First build takes longer (compiling CoreML model)
- No code changes required
