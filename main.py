import json  # Para formatar o payload JSON ao enviar arquivos
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import requests  # Para requisições HTTP
import torch
from dotenv import load_dotenv
from PIL import Image, ImageTk
from ultralytics import YOLO

load_dotenv()

# --- Configurações ---
# (Seções AVAILABLE_MODELS, AVAILABLE_SOURCES, DEFAULT_SOURCE_NAME, CONFIDENCE_THRESHOLD, TARGET_CLASSES permanecem as mesmas)
AVAILABLE_MODELS = {
    "YOLOv11n - Sem finetuning": "yolo11n.pt",
    "YOLOv11s - Sem finetuning": "yolo11s.pt",
    "YOLOv11n - COCO": "yolo11n_dataset_coco.pt",
    "YOLOv11n - Knife - Scissors": "yolo11n_dataset_knife_scissors.pt",
    "YOLOv11s - Knife - Scissors": "yolo11s_dataset_knife_scissors.pt",
    "YOLOv11s - All Knives": "yolo11s_dataset_knife.pt",
    "YOLOv11m - Sem finetuning": "yolo11m.pt",
    # Adicione mais modelos aqui se necessário
    # "Outro Modelo": "caminho/para/outro_modelo.pt"
}

# Dicionário de fontes de vídeo disponíveis (Nome Amigável: Valor para OpenCV)
AVAILABLE_SOURCES = {
    "Webcam Interna (Mobile)": 0,
    "Webcam Interna (PC)": 1,
    "Câmera IP": "rtsp://admin:Teste123@192.168.15.59:554/onvif1",
    "Video 1 (0:42)": "data/inputs/video.mp4",
    "Video 2 (0:04)": "data/inputs/video2.mp4",
    # Adicione outras câmeras ou arquivos de vídeo aqui
    # "Vídeo Arquivo": "caminho/para/video.mp4"
}
DEFAULT_SOURCE_NAME = "Webcam Externa (1)"
CONFIDENCE_THRESHOLD = 0.40
TARGET_CLASSES = ["knife", "scissors"]

# --- Configurações de Alerta Discord ---
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
ALERT_CLASSES = ["knife"]
ALERT_THRESHOLD = 0.70
ALERT_COOLDOWN_SECONDS = 60
# --- Fim das Configurações ---

# (Variáveis Globais permanecem as mesmas)
cap = None
current_model = None
current_source_value = None
model_lock = threading.Lock()
capture_lock = threading.Lock()
root = None
video_label = None
fps_start_time = 0
fps_frame_count = 0
fps_display = 0.0
running = True
last_alert_time = 0

# --- Funções ---


def send_discord_alert(detected_class, confidence_level, image_frame):
    """Envia uma mensagem de alerta COM IMAGEM para o webhook do Discord."""
    global DISCORD_WEBHOOK_URL
    if not DISCORD_WEBHOOK_URL:
        print("AVISO: URL do Webhook do Discord não configurada.")
        return

    message_content = f"🚨 ALERTA! Objeto detectado: **{detected_class.upper()}** com confiança **{confidence_level:.2f}**"
    payload_json = {"content": message_content}

    # Codificar o frame (NumPy array) para PNG em memória
    try:
        ret, buffer = cv2.imencode(".png", image_frame)
        if not ret:
            print("Erro ao codificar a imagem para envio.")
            # Tentar enviar apenas a mensagem de texto como fallback?
            # requests.post(DISCORD_WEBHOOK_URL, json=payload_json, timeout=10)
            return
        image_bytes = buffer.tobytes()
    except Exception as e:
        print(f"Erro durante a codificação da imagem: {e}")
        return

    # Preparar os dados para multipart/form-data
    files = {
        # O payload JSON vai como um campo de formulário
        "payload_json": (None, json.dumps(payload_json), "application/json"),
        # O arquivo de imagem
        # 'file' é um nome comum, mas Discord pode aceitar outros como 'files[0]'
        "file": ("detection.png", image_bytes, "image/png"),
    }

    try:
        # Enviar a requisição POST com 'files' em vez de 'json'
        response = requests.post(
            DISCORD_WEBHOOK_URL, files=files, timeout=15
        )  # Aumentar timeout para upload
        response.raise_for_status()
        print(
            f"Mensagem de alerta e imagem '{detected_class}' enviadas com sucesso para o Discord!"
        )
    except requests.exceptions.RequestException as e:
        print(f"Erro ao enviar mensagem/imagem para o Discord: {e}")
        if response is not None:
            print(
                f"Resposta do servidor Discord: {response.text}"
            )  # Logar resposta em caso de erro
    except Exception as e:
        print(f"Erro inesperado ao enviar para o Discord: {e}")


# (load_model, change_video_source, on_model_select, on_source_select permanecem as mesmas)
def load_model(model_path):
    global current_model
    try:
        print(f"Carregando modelo: {model_path}...")
        new_model = YOLO(model_path)
        if hasattr(new_model, "names"):
            with model_lock:
                current_model = new_model
            print(f"Modelo {model_path} carregado.")
        else:
            print(f"Erro: {model_path} não é um modelo YOLO válido.")
    except Exception as e:
        print(f"Erro ao carregar modelo {model_path}: {e}")


def change_video_source(source_value):
    global cap, fps_start_time, fps_frame_count, current_source_value
    print(f"Trocando para fonte: {source_value}")
    with capture_lock:
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None
        print(f"Abrindo fonte: {source_value}")
        new_cap = cv2.VideoCapture(source_value)
        if new_cap.isOpened():
            cap = new_cap
            current_source_value = source_value
            print("Fonte aberta.")
            fps_start_time = time.time()
            fps_frame_count = 0
        else:
            print(f"Erro ao abrir fonte {source_value}")
            cap = None


def on_model_select(event=None):
    selected_path = "models/" + AVAILABLE_MODELS.get(model_combobox.get())
    if selected_path:
        threading.Thread(target=load_model, args=(selected_path,), daemon=True).start()
    else:
        print("Erro: Modelo não encontrado.")


def on_source_select(event=None):
    selected_value = AVAILABLE_SOURCES.get(source_combobox.get())
    if selected_value is not None:
        change_video_source(selected_value)
    else:
        print("Erro: Fonte não encontrada.")


def update_frame():
    """Captura um frame, processa, atualiza a interface e envia alertas."""
    global fps_start_time, fps_frame_count, fps_display, running, current_model, cap, last_alert_time

    if not running:
        return

    frame = None
    processed_frame = None

    with capture_lock:
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = None

    if frame is None:
        if running:
            root.after(30, update_frame)
        return

    # --- Processamento ---
    processed_frame = frame.copy()  # Importante trabalhar na cópia para desenhar

    # --- Cálculo FPS ---
    fps_frame_count += 1
    current_time_fps = time.time()
    elapsed_time = current_time_fps - fps_start_time
    if elapsed_time >= 1.0:
        fps_display = fps_frame_count / elapsed_time
        fps_start_time = current_time_fps
        fps_frame_count = 0

    # --- Detecção YOLO ---
    model_to_use = None
    with model_lock:
        model_to_use = current_model

    detection_occurred_in_frame = False

    if model_to_use:
        try:
            results = model_to_use(frame, verbose=False)  # Detectar no frame original

            for result in results:
                boxes = result.boxes
                if not hasattr(result, "names"):
                    continue

                for box in boxes:
                    detection_occurred_in_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    if class_id < len(result.names):
                        class_name = result.names[class_id]
                        class_name_lower = class_name.lower()

                        # Desenhar caixa
                        if (
                            class_name_lower in TARGET_CLASSES
                            and confidence > CONFIDENCE_THRESHOLD
                        ):
                            colorAlarme = (0, 0, 255)
                            if confidence > 0.5:
                                colorAlarme = (0, 255, 0)
                            # Desenhar no processed_frame
                            cv2.rectangle(
                                processed_frame, (x1, y1), (x2, y2), colorAlarme, 2
                            )
                            label_text = f"{class_name} {confidence:.2f}"
                            cv2.putText(
                                processed_frame,
                                label_text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                colorAlarme,
                                2,
                            )

                        # --- Lógica de Alerta Discord com Imagem ---
                        if (
                            class_name_lower in ALERT_CLASSES
                            and confidence > ALERT_THRESHOLD
                        ):
                            current_time_alert = time.time()
                            if (
                                current_time_alert - last_alert_time
                                >= ALERT_COOLDOWN_SECONDS
                            ):
                                print(
                                    f"--- ALERTA DETECTADO ({time.strftime('%H:%M:%S')}) ---"
                                )
                                print(
                                    f"Objeto: {class_name}, Confiança: {confidence:.2f}, Enviando com imagem..."
                                )

                                # *** Passar uma CÓPIA do frame processado para a thread ***
                                frame_copy_for_alert = processed_frame.copy()

                                threading.Thread(
                                    target=send_discord_alert,
                                    args=(class_name, confidence, frame_copy_for_alert),
                                    daemon=True,
                                ).start()

                                last_alert_time = current_time_alert
                            # else: # Log de cooldown (opcional)
                            # time_remaining = ALERT_COOLDOWN_SECONDS - (current_time_alert - last_alert_time)
                            # print(f"Alerta ({class_name_lower}) suprimido. Cooldown: {time_remaining:.1f}s.")
                    else:
                        print(f"Aviso: ID de classe {class_id} inválido.")

        except Exception as e:
            print(f"Erro durante a inferência YOLO: {e}")

    # --- Exibir FPS ---
    # Desenhar no processed_frame
    fps_text = f"FPS: {fps_display:.2f}"
    (h, w) = processed_frame.shape[:2]
    text_x = w - 150
    text_y = h - 20
    cv2.putText(
        processed_frame,
        fps_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # --- Exibir na Interface Tkinter ---
    try:
        cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    except Exception as e:
        print(f"Erro ao atualizar Tkinter: {e}")

    # Agendar próxima atualização
    if running:
        delay_ms = 15 if not detection_occurred_in_frame else 25
        root.after(delay_ms, update_frame)


def on_closing():
    """Função chamada ao fechar a janela Tkinter."""
    global running, cap
    print("Fechando...")
    running = False
    time.sleep(0.2)
    with capture_lock:
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None
            print("Webcam liberada.")
    if root is not None:
        try:
            root.destroy()
            print("Janela Tkinter destruída.")
        except tk.TclError:
            pass
    cv2.destroyAllWindows()


# --- Inicialização ---
if __name__ == "__main__":
    # (Inicialização da fonte de vídeo)
    initial_source_value = AVAILABLE_SOURCES.get(DEFAULT_SOURCE_NAME)
    if initial_source_value is None:
        print(f"Erro: Fonte padrão '{DEFAULT_SOURCE_NAME}' não encontrada.")
        if AVAILABLE_SOURCES:
            DEFAULT_SOURCE_NAME = list(AVAILABLE_SOURCES.keys())[0]
            initial_source_value = AVAILABLE_SOURCES[DEFAULT_SOURCE_NAME]
            print(f"Usando primeira fonte: '{DEFAULT_SOURCE_NAME}'")
        else:
            exit("Erro: Nenhuma fonte de vídeo definida.")
    print(f"Abrindo fonte: {initial_source_value} ({DEFAULT_SOURCE_NAME})")
    cap = cv2.VideoCapture(initial_source_value)
    if not cap.isOpened():
        exit(f"Erro Crítico: Não foi possível abrir {initial_source_value}")
    print("Fonte inicial aberta.")
    current_source_value = initial_source_value

    # (Carregamento do Modelo Inicial)
    if AVAILABLE_MODELS:
        initial_model_name = list(AVAILABLE_MODELS.keys())[0]
        initial_model_path = AVAILABLE_MODELS[initial_model_name]
        load_model(initial_model_path)
    else:
        print("Aviso: Nenhum modelo definido.")
        initial_model_name = "Nenhum Modelo"

    # (Configuração da GUI Tkinter - igual à anterior)
    root = tk.Tk()
    root.title("Detecção com Alerta Discord (Imagem)")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10, padx=10, fill=tk.X)
    tk.Label(top_frame, text="Modelo:").pack(side=tk.LEFT, padx=(0, 5))
    model_combobox = ttk.Combobox(
        top_frame, values=list(AVAILABLE_MODELS.keys()), state="readonly", width=25
    )
    if AVAILABLE_MODELS:
        model_combobox.set(initial_model_name)
    model_combobox.bind("<<ComboboxSelected>>", on_model_select)
    model_combobox.pack(side=tk.LEFT, padx=(0, 15))
    tk.Label(top_frame, text="Fonte:").pack(side=tk.LEFT, padx=(0, 5))
    source_combobox = ttk.Combobox(
        top_frame, values=list(AVAILABLE_SOURCES.keys()), state="readonly", width=25
    )
    source_combobox.set(DEFAULT_SOURCE_NAME)
    source_combobox.bind("<<ComboboxSelected>>", on_source_select)
    source_combobox.pack(side=tk.LEFT, padx=(0, 5))
    video_label = tk.Label(root)
    video_label.pack(padx=10, pady=(0, 10))

    # --- Iniciar ---
    fps_start_time = time.time()
    last_alert_time = 0
    root.after(100, update_frame)
    root.mainloop()

    print("Programa encerrado.")
