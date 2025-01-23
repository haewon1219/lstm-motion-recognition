import cv2
import mediapipe as mp
import numpy as np
import torch

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # 관절 시각화를 위한 유틸리티

# 웹캠 열기
cap = cv2.VideoCapture(0)

# LSTM 모델 불러오기
model_path = "C:/Users/Admin/Desktop/lstm_model_scripted.pt"  # 사전 저장된 LSTM 모델 경로
model = torch.jit.load(model_path)
model.eval()

# 라벨 매핑
labels_map = {0: "Back", 1: "Squat", 2: "Side"}

# 관절 데이터 저장 변수
is_collecting = False  # 관절 추출 상태 플래그
keypoints_list = []  # 관절 데이터 저장

# 표준화 함수
def standardize_data(data):
    """
    데이터 표준화 (평균 0, 표준편차 1).
    Args:
        data (np.array): 원본 데이터, Shape: (samples, frames, joints, coords).
    Returns:
        np.array: 표준화된 데이터.
    """
    mean = data.mean(axis=(1, 2, 3), keepdims=True)  # 샘플별 평균 계산
    std = data.std(axis=(1, 2, 3), keepdims=True)  # 샘플별 표준편차 계산
    standardized_data = (data - mean) / (std + 1e-8)  # 표준화
    return standardized_data

# 실시간 예측 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe로 관절 데이터 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)

    # 관절 데이터 추출 및 시각화
    if result.pose_landmarks:
        # Mediapipe 유틸리티로 관절 시각화
        mp_drawing.draw_landmarks(
            frame,  
            result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  # 관절 연결선
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # 관절점 스타일
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 연결선 스타일
        )

        # 33개의 모든 관절 좌표 추출 (x, y, z)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])

        # 관절 추출 상태에서 데이터 저장
        if is_collecting:
            keypoints_list.append(keypoints)

        # 관절 추출 상태 표시
        status_text = "Collecting" if is_collecting else "Paused"
        cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 영상 출력
    cv2.imshow("Live Feed", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q'를 누르면 종료
        break
    elif key == ord(' '):  # 스페이스바로 상태 토글
        is_collecting = not is_collecting
        if not is_collecting and keypoints_list:  # 관절 추출 종료 시 모델에 데이터 입력
            # 쌓인 관절 데이터를 배열로 변환
            keypoints_array = np.array(keypoints_list)  # (시퀀스 길이, 33, 3)
            
            # 데이터 정규화
            keypoints_standardized = standardize_data(np.expand_dims(keypoints_array, axis=0))[0]  # 정규화 후 원래 형태로 복원

            # 모델 입력 형태 변환 (배치, 시퀀스 길이, 관절 수 * 좌표 수)
            input_data = torch.tensor(keypoints_standardized).float()
            input_data = input_data.view(1, keypoints_standardized.shape[0], -1)  # (1, 시퀀스 길이, 33*3)

            # 모델 예측
            with torch.no_grad():
                outputs = model(input_data)  # 모델 출력: logits
                probabilities = torch.softmax(outputs, dim=1)  # 확률로 변환
                predicted_class = torch.argmax(probabilities, dim=1).item()  # 예측 클래스
                confidence = probabilities[0, predicted_class].item()  # 예측 확률

            # 결과 출력
            predicted_label = labels_map[predicted_class]
            confidence_percent = confidence * 100  # 퍼센트로 변환
            print(f"Predicted: {predicted_label}, Confidence: {confidence_percent:.2f}%")

            # 각 동작별 확률 출력
            print("Probabilities:")
            for i, prob in enumerate(probabilities[0]):
                label = labels_map[i]
                print(f"  {label}: {prob.item() * 100:.2f}%")

            # 저장된 관절 데이터 초기화
            keypoints_list = []

# 종료
cap.release()
cv2.destroyAllWindows() 
