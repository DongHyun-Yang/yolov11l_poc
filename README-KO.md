# YOLOv11 Object Detection Project

이 프로젝트는 YOLOv11 모델을 사용하여 객체 탐지를 수행하는 Python 기반 애플리케이션입니다. PyTorch 모델을 ONNX 형식으로 변환하고, ONNX Runtime을 사용하여 고성능 추론을 제공합니다.

## 🗂️ 디렉토리 구조

```plaintext
yolov11l_test/
├── README.md
├── test_images/               # 테스트용 이미지 폴더
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── yolov11n/                  # YOLOv11 nano 모델
│   ├── export.py              # 모델 변환 스크립트
│   ├── main.py                # 객체 탐지 실행 스크립트
│   ├── models/                # 모델 파일 저장 폴더
│   │   ├── yolo11l.pt         # PyTorch 모델 파일
│   │   └── yolo11l.onnx       # 변환된 ONNX 모델 파일
│   └── output/                # 탐지 결과 이미지 저장 폴더
│       └── *_detected.jpg
└── (추후 추가 예정)
    ├── yolo11l-pose/          # YOLOv11 포즈 추정
    ├── yolo11l-seg/           # YOLOv11 인스턴스 분할
    └── yolo11l-obb/           # YOLOv11 지향성 경계 상자
```

## 🛠️ 사전 준비사항

### 1. Python 환경 요구사항

- Python 3.8 이상

### 2. 필요한 패키지 설치

```bash
# 기본 패키지 설치
pip install ultralytics opencv-python numpy onnxruntime pathlib

# 또는 requirements.txt를 사용하는 경우
pip install -r requirements.txt
```

### 3. 주요 의존성 패키지

- **ultralytics**: YOLOv11 모델 로드 및 변환
- **opencv-python**: 이미지 처리 및 시각화
- **numpy**: 수치 연산
- **onnxruntime**: ONNX 모델 추론
- **pathlib**: 파일 경로 처리

## 📥 모델 다운로드

### YOLOv11 모델 다운로드

YOLOv11 모델은 [Ultralytics 공식 문서](https://docs.ultralytics.com/ko/models/yolo11/#performance-metrics)에서 다운로드할 수 있습니다.

```bash
# yolo11n 폴더로 이동
cd yolov11n

# models 폴더 생성
mkdir models

# YOLOv11l 모델 다운로드 (약 50MB)
# 다음 중 하나의 방법을 사용:

# 방법 1: wget 사용
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -O models/yolo11l.pt

# 방법 2: 직접 다운로드
# 브라우저에서 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt 를 다운로드하여 models/ 폴더에 저장
```

## 🚀 사용 방법

### 1. 모델 변환 (PyTorch → ONNX)

```bash
cd yolov11n
python export.py
```

**export.py 실행 결과:**

- `models/yolo11l.pt` → `models/yolo11l.onnx` 변환
- SSL 인증서 문제 자동 해결
- 오프라인 모드 지원
- ONNX opset 12 사용으로 최대 호환성 보장

### 2. 객체 탐지 실행

```bash
cd yolov11n
python main.py
```

**main.py 실행 과정:**

1. `../test_images/` 폴더에서 모든 `*.jpg`, `*.jpeg` 파일 검색
2. 각 이미지에 대해 객체 탐지 수행
3. 탐지 결과를 `output/{파일명}_detected.jpg` 형식으로 저장
4. 콘솔에 탐지된 객체 정보 출력

### 3. 실행 결과 예시

```plaintext
총 9개의 이미지 파일을 처리합니다.
결과는 'output' 폴더에 저장됩니다.
--------------------------------------------------

[1/9] 처리 중: 1.jpg
탐지 결과가 'output\1_detected.jpg' 파일로 저장되었습니다.
[1] 총 8개의 객체가 탐지되었습니다.
  1. person: 0.91 - 위치: (385, 69) ~ (499, 347)
  2. bowl: 0.81 - 위치: (30, 343) ~ (99, 384)
  3. spoon: 0.76 - 위치: (531, 43) ~ (553, 126)
  ...

--------------------------------------------------
처리 완료: 9/9개 성공
결과 파일들이 'output' 폴더에 저장되었습니다.
```

## 📊 지원되는 객체 클래스

YOLOv11은 COCO 데이터셋으로 훈련되어 80개 클래스의 객체를 탐지할 수 있습니다:

**사람 및 동물**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**교통수단**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**생활용품**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**가구 및 전자제품**: chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**기타**: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## ⚙️ 설정 옵션

### main.py 설정 변경

```python
# 모델 및 경로 설정
ONNX_MODEL_PATH = 'models/yolo11l.onnx'  # ONNX 모델 경로
IMAGE_PATH = '../test_images/'            # 입력 이미지 경로 (파일 또는 폴더)
OUTPUT_DIR = 'output'                     # 결과 저장 폴더

# 탐지 임계값 설정
CONFIDENCE_THRESHOLD = 0.5    # 신뢰도 임계값 (0.0 ~ 1.0)
IOU_THRESHOLD = 0.5          # IoU 임계값 (중복 제거)
SCORE_THRESHOLD = 0.5        # 점수 임계값
```

## 🔧 문제 해결

### SSL 인증서 오류

- 회사 방화벽이나 프록시 환경에서 SSL 인증서 오류가 발생할 수 있습니다
- `export.py`에 SSL 우회 코드가 포함되어 있어 자동으로 해결됩니다

### ONNX Runtime 오류

```bash
# CPU 버전 재설치
pip uninstall onnxruntime
pip install onnxruntime

# GPU 버전 사용 시 (NVIDIA GPU 필요)
pip install onnxruntime-gpu
```

### 메모리 부족 오류

- 큰 이미지나 많은 수의 이미지 처리 시 메모리 부족이 발생할 수 있습니다
- 배치 크기를 줄이거나 이미지 크기를 축소해서 처리하세요

## 📈 성능 정보

- **모델 크기**: YOLOv11l (약 50MB)
- **추론 속도**: CPU에서 이미지당 약 200-500ms
- **지원 해상도**: 최대 640x640 (자동 리사이징)
- **정확도**: mAP50-95: 53.9% (COCO val2017 기준)

## 🚧 향후 계획

- **yolo11l-pose/**: 포즈 추정 기능 추가
- **yolo11l-seg/**: 인스턴스 분할 기능 추가
- **yolo11l-obb/**: 지향성 경계 상자 탐지 추가
- 웹 인터페이스 추가
- 실시간 비디오 처리 기능
- 모바일 최적화 버전

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. YOLOv11 모델은 [Ultralytics](https://ultralytics.com/) 라이선스를 따릅니다.

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

---

### Created with ❤️ using YOLOv11 and ONNX Runtime
