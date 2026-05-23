import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import threading
import time

# ------------------------------
# CONFIG
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fusion weights
PIC_WEIGHT = 0.25
TAB_WEIGHT = 0.75

# Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------
# LOAD PICTORIAL MODEL (ResNet50)
# ------------------------------
NUM_CLASSES = 7
resnet_model = models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model.load_state_dict(torch.load("best_rafdb_resnet50.pth", map_location=DEVICE))
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

# ------------------------------
# LOAD TABULAR MODEL
# ------------------------------
tab_state = torch.load("student_stress_model.pth", map_location="cpu")

class TabularNN:
    def __init__(self, state):
        self.w1 = state["w1"].numpy() if isinstance(state["w1"], torch.Tensor) else state["w1"]
        self.b1 = state["b1"].numpy() if isinstance(state["b1"], torch.Tensor) else state["b1"]
        self.w2 = state["w2"].numpy() if isinstance(state["w2"], torch.Tensor) else state["w2"]
        self.b2 = state["b2"].numpy() if isinstance(state["b2"], torch.Tensor) else state["b2"]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        return self.sigmoid(z2)

tab_model = TabularNN(tab_state)

# Scaling info
dataset_medians = tab_state.get("medians", np.zeros(tab_state["w1"].shape[0]))
dataset_std_devs = tab_state.get("stds", np.ones(tab_state["w1"].shape[0]))

def scale_tabular(x):
    scaled = (x - dataset_medians) / (dataset_std_devs + 1e-8)
    return np.clip(scaled, -3.0, 3.0)

# ------------------------------
# SHARED VARIABLES
# ------------------------------
emotion_scores = []  # Running list of emotion scores from camera
tab_score = None     # Will be set after survey
stop_camera = False  # Signal to stop camera thread

# ------------------------------
# CAMERA THREAD
# ------------------------------
def camera_loop():
    global emotion_scores, tab_score, stop_camera

    camera_url = "http://192.168.0.103:4747/video"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        stop_camera = True
        return

    emotion_map = {
        0: "Anger",
        1: "Disgust",
        2: "Fear",
        3: "Happiness",
        4: "Sadness",
        5: "Surprise",
        6: "Neutral"
    }

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        emotions_in_frame = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            face_tensor = torch.tensor(face_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
            face_tensor = face_tensor.to(DEVICE)

            with torch.no_grad():
                out = resnet_model(face_tensor)
                pred_class = out.argmax(dim=1).item()
                emotions_in_frame.append(pred_class)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            if emotions_in_frame:
                cv2.putText(frame, emotion_map[pred_class], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if emotions_in_frame:
            avg_emotion = np.mean(emotions_in_frame) / (NUM_CLASSES-1) * 100
            emotion_scores.append(avg_emotion)
            avg_emotion_score = np.mean(emotion_scores)
        else:
            avg_emotion_score = 0

        # Compute fused stress score if tab_score is available
        if tab_score is not None:
            final_stress = PIC_WEIGHT * avg_emotion_score + TAB_WEIGHT * tab_score
            cv2.putText(frame, f"Stress Level: {final_stress:.2f}/100", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("Stress Fusion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera = True
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# SURVEY INPUT (TABULAR)
# ------------------------------
def run_survey():
    global tab_score
    print("\n--- Personal Stress Level Survey ---")
    features = tab_state.get("features", [f"Feature_{i}" for i in range(tab_state["w1"].shape[0])])
    responses = []

    for i, f_name in enumerate(features):
        try:
            val = float(input(f"{f_name}: "))
            responses.append(val)
        except:
            responses.append(dataset_medians[i])

    test_input = scale_tabular(np.array(responses).reshape(1,-1))
    tab_score = tab_model.forward(test_input)[0][0] * 100
    print(f"\nTabular stress score: {tab_score:.2f}/100")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop)
    cam_thread.start()

    # Run survey on main thread
    run_survey()

    print("Survey complete! Camera fusion will continue. Press 'Q' on window to exit.")

    # Wait for camera thread to finish
    cam_thread.join()
