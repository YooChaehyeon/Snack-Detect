import cv2
from ultralytics import YOLO

def main():
    model_path = "runs/detect/snack_24_species/weights/best.pt"
    print(f"🧠 초경량 24종 가중치 로드 중: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 웹캠이나 가상 카메라 설정을 확인해 주세요.")
        return

    # 윈도우 창 생성 및 항상 맨 위에 오도록 설정
    window_name = "Snack 24 Real-Time Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("\n🚀 실시간 과자 인식 카메라 시작 (멀티 컬러 ROI 모드)!")
    print("💡 종료하려면 카메라 창을 선택하고 키보드 'q'를 누르세요.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ❗️ conf=0.10 으로 낮춰서 10%의 가능성만 보여도 무조건 박스를 그리도록 강제합니다.
        results = model(frame, conf=0.10, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                class_name = model.names[cls_idx]

                # 💡 [핵심 추가] 과자 종류(cls_idx)에 따라 고유한 고정 색상(BGR)을 수학적으로 계산합니다.
                # 클래스 번호마다 고유한 숫자가 배정되므로, 과자 종류마다 무조건 다른 색상이 맺힙니다.
                # 너무 어두운 색이 나오는 것을 방지하기 위해 각 채널에 50을 더하고 255로 나머지 연산을 합니다.
                b = (cls_idx * 45 + 100) % 256
                g = (cls_idx * 85 + 50) % 256
                r_color = (cls_idx * 125 + 150) % 256
                box_color = (b, g, r_color) # OpenCV에서 쓰는 BGR 튜플 형태

                # 💡 지정된 고유 색상(box_color)으로 박스와 글씨를 표기합니다.
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                label_text = f"{class_name} {conf*100:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 카메라 테스트가 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()