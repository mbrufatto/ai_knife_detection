# Requer: pip install ultralytics opencv-python tqdm

import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm
from collections import deque

# --- CONFIGURAÇÕES ---
input_video_path = '../data/input/video.mp4'  # Caminho do vídeo de entrada
output_video_path = '../data/output/output_detected.mp4'  # Caminho do vídeo de saída
model_path = '../models/yolo11s_dataset_knife_scissors.pt'  # Caminho para o modelo YOLO treinado

try:
    # --- CARREGAR MODELO ---
    print("🔄 Carregando modelo YOLO...")
    model = YOLO(model_path)

    # --- ABRIR VÍDEO DE ENTRADA ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- CONFIGURAR VÍDEO DE SAÍDA ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- CONFIGURAR HISTÓRICO DE DETECÇÕES ---
    detection_history = {"fork": deque(maxlen=3), "knife": deque(maxlen=3), "scissors": deque(maxlen=3)}

    print("▶️ Iniciando processamento de vídeo...")
    with tqdm(total=frame_count, desc="Processando frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            alert_triggered = False
            alert_frame = None

            # Realiza a predição com confiança mínima de 0.25
            results = model.predict(frame, conf=0.25, verbose=False)

            # Coletar classes detectadas neste frame
            current_detections = []

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
                classes = r.boxes.cls.cpu().numpy() if r.boxes else []
                confs = r.boxes.conf.cpu().numpy() if r.boxes else []
                names = r.names

                for box, cls, conf in zip(boxes, classes, confs):
                    label = names[int(cls)]
                    if label in detection_history:
                        current_detections.append((label, box, conf))

            # Atualizar histórico de detecção por classe
            for class_name in detection_history:
                detection_history[class_name].append(0)  # começa com 0

            for label, box, conf in current_detections:
                detection_history[label][-1] = 1  # marca presença no histórico

            # Verificar se alguma classe foi detectada em 3 frames consecutivos
            confirmed_class = None
            for class_name, history in detection_history.items():
                if sum(history) == 3:
                    confirmed_class = class_name
                    break

            # Desenhar os resultados e decidir se alerta será gerado
            for label, box, conf in current_detections:
                x1, y1, x2, y2 = map(int, box)
                if label == confirmed_class and conf >= 0.25:
                    alert_triggered = True
                    alert_frame = frame.copy()
                    color = (0, 0, 255)  # vermelho para alerta
                else:
                    color = (0, 255, 0)  # verde padrão

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Escreve o frame normal
            out.write(frame)
            pbar.update(1)


    # --- FINALIZAR ---
    cap.release()
    out.release()
    print(f"✅ Vídeo salvo em: {os.path.abspath(output_video_path)}")

except Exception as e:
    print(f"❌ Ocorreu um erro: {str(e)}")
