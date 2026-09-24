import os
from roboflow import Roboflow
from ultralytics import YOLO


def main():
    print("생성 완료된 24종 과자 데이터셋 다운로드를 시작합니다...")

    # API 키는 코드에 저장하지 않고 환경변수에서 불러옵니다.
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY 환경변수를 설정해주세요.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("-yzlpj").project("snack-dlgc5-siwhv")
    version = project.version(1)
    dataset = version.download("yolov8")

    yaml_path = os.path.join(dataset.location, "data.yaml")
    print(f"✅ 24종 데이터셋 다운로드 완료! 설정 파일 위치: {yaml_path}")

    print("과자 분류용 AI 모델 학습을 시작합니다. 잠시만 기다려주세요...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        device="cpu",
        name="snack_24_species",
    )
    print("과자 모델 학습이 최종 완료되었습니다")


if __name__ == "__main__":
    main()
