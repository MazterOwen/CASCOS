import time
from collections import Counter

import cv2
import torch
import yaml
from ultralytics import YOLO

import pluma

# ---------------- CONFIGURACIÓN ----------------
RUTA_MODELO = r"C:\Users\PC\PycharmProjects\CASCOS\runs\detect\train\weights\best.pt"
RUTA_DATASET = r"C:\Users\PC\Documents\DATASETcasco\motociclista-con-y-sin-casco3.v1i.yolo26"
# Fuente de video:
#   0, 1, 2...                       -> cámara USB / webcam
#   "rtsp://usuario:clave@IP:554/..." -> cámara IP (RTSP)
#   "http://IP:puerto/video"          -> cámara IP (HTTP / celular con app tipo IP Webcam)
CAMARA = pluma.elegir_camara()  # muestra las cámaras conectadas y pide elegir una
CONF_MINIMA = 0.5          # confianza mínima de detección
IMGSZ = 640                # tamaño de entrada (igual que en el entrenamiento)
FRAMES_CONFIRMAR = 5       # frames seguidos necesarios para tomar una decisión (evita parpadeos)
SEGUNDOS_PLUMA_ABIERTA = 5 # tiempo que la pluma permanece levantada
USAR_ARDUINO = True        # False para probar sin Arduino

# ---------------- CARGA DEL MODELO Y DATASET ----------------
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Dispositivo: {'GPU' if device == 0 else 'CPU'}")

with open(f"{RUTA_DATASET}\\data.yaml", encoding="utf-8") as f:
    datos = yaml.safe_load(f)
nombres_dataset = datos["names"]
print(f"[INFO] Clases del dataset: {nombres_dataset}")

model = YOLO(RUTA_MODELO)
model.fuse()  # optimiza la red para inferencia más rápida

CON_CASCO = "casco"       # nombre de la clase en best.pt
SIN_CASCO = "no-casco"    # nombre de la clase en best.pt


def obtener_estado(result):
    """Devuelve 'SIN_CASCO', 'CON_CASCO' o None según lo detectado en el frame."""
    detectados = {model.names[int(c)] for c in result.boxes.cls}
    if SIN_CASCO in detectados:
        return "SIN_CASCO"   # si hay alguien sin casco, se prioriza la alerta
    if CON_CASCO in detectados:
        return "CON_CASCO"
    return None


def main():
    if USAR_ARDUINO:
        pluma.conectar()

    if isinstance(CAMARA, int):
        # Cámara local: en Windows CAP_DSHOW abre más rápido
        cap = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        # Cámara IP: buffer pequeño para evitar retraso en el video
        cap = cv2.VideoCapture(CAMARA)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara: {CAMARA}")
        return
    print("[INFO] Cámara conectada. Presiona 'q' para salir.")

    historial = []           # últimos estados para confirmar la decisión
    estado_actual = None     # decisión confirmada
    pluma_abierta = False
    tiempo_apertura = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Se perdió la señal de la cámara.")
            break

        results = model.predict(
            frame, conf=CONF_MINIMA, imgsz=IMGSZ, device=device,
            half=(device == 0), verbose=False
        )
        result = results[0]

        # Estado del frame y confirmación por varios frames seguidos
        historial.append(obtener_estado(result))
        historial = historial[-FRAMES_CONFIRMAR:]
        conteo = Counter(historial)
        estado_frame, veces = conteo.most_common(1)[0]
        if estado_frame is not None and veces >= FRAMES_CONFIRMAR and estado_frame != estado_actual:
            estado_actual = estado_frame
            if estado_actual == "CON_CASCO":
                print("[OK] Motociclista con casco. Puede pasar.")
                if USAR_ARDUINO and not pluma_abierta:
                    pluma.abrir()
                    pluma_abierta = True
                    tiempo_apertura = time.time()
            elif estado_actual == "SIN_CASCO":
                print("[ALERTA] Persona sin casco")
                if USAR_ARDUINO and pluma_abierta:
                    pluma.cerrar()
                    pluma_abierta = False

        # Cerrar la pluma automáticamente después del tiempo configurado
        if pluma_abierta and time.time() - tiempo_apertura > SEGUNDOS_PLUMA_ABIERTA:
            if USAR_ARDUINO:
                pluma.cerrar()
            pluma_abierta = False

        # Dibujar cajas
        for box in result.boxes:
            nombre = model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if nombre == CON_CASCO else (0, 0, 255) if nombre == SIN_CASCO else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{nombre} {float(box.conf):.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mensaje principal en pantalla
        if estado_actual == "SIN_CASCO":
            cv2.putText(frame, "PERSONA SIN CASCO", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif estado_actual == "CON_CASCO":
            cv2.putText(frame, "CON CASCO - PUEDE PASAR", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Control de casco", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if USAR_ARDUINO:
        pluma.cerrar()
        pluma.desconectar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
