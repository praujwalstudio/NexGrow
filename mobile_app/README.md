# NexGrow Mobile App

Flutter-based mobile UI for on-device cotton leaf disease diagnosis.

## Setup

```bash
cd mobile_app
flutter pub get
flutter run
```

## Architecture

- Uses INT8 quantized ONNX model for offline inference
- Camera integration for real-time leaf capture
- Grad-CAM++ visualization overlay
- Weather-aware treatment protocol display

## Requirements

- Flutter 3.x
- Dart 3.x
- Android SDK / iOS Xcode
