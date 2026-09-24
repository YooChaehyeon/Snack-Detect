# Snack Detect

YOLOv8로 웹캠 화면에서 24종의 과자를 실시간으로 인식하는 프로젝트입니다.

학습 완료 모델 `best.pt`가 저장소에 포함되어 있으므로 원본 학습 데이터가 없어도 바로 과자 인식을 실행할 수 있습니다.

## 다시 내려받아 실행하기

1. 저장소를 내려받아 VS Code로 폴더를 엽니다.
2. Python 3.10 또는 3.11 환경을 준비합니다.
3. VS Code 터미널에서 다음 명령을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python detect_cam.py
```

카메라 창에서 `q`를 누르면 종료됩니다. 기본 카메라는 장치 번호 0을 사용합니다.

## 모델 다시 학습하기

새로 학습하려면 개인 Roboflow API 키를 `ROBOFLOW_API_KEY` 환경 변수로 설정한 뒤 아래 명령을 실행합니다. API 키는 공개 저장소에 올리지 마세요.

```powershell
$env:ROBOFLOW_API_KEY="본인의_API_키"
python snack_v2.py
```

학습이 끝나면 `runs/detect/snack_24_species/weights/best.pt`를 저장소 루트의 `best.pt`로 복사하면 새 모델을 사용할 수 있습니다.
