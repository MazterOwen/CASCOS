import queue
import threading
import time
import tkinter as tk
from collections import Counter
import os
from tkinter import ttk, scrolledtext, filedialog

import cv2
import torch
import yaml
from PIL import Image, ImageTk
from ultralytics import YOLO

import pluma


# ---------------- CONFIGURACIÓN FIJA ----------------

RUTA_MODELO = r"C:\Users\PC\PycharmProjects\CASCOS\runs\detect\entrenamientos\detector_con_casco_ingelam_rapido\weights\best.pt"

RUTA_DATASET = r"C:\Users\PC\Documents\DATASETcasco\Ingelam.v1i.yolo26"

IMGSZ = 416

CON_CASCO = "casco"       # nombre de la clase en best.pt
SIN_CASCO = "no-casco"    # nombre de la clase en best.pt

# Paleta (parecida al estilo del dashboard de referencia)
FONDO = "#F5F1EA"
TARJETA = "#FFFFFF"
BORDE = "#D9D2C5"
TEXTO = "#1A1F2E"
TEXTO_SUAVE = "#6B7280"
VERDE = "#16A34A"
ROJO = "#DC2626"
AZUL = "#0EA5E9"
GRIS_BTN = "#ECE7DC"
VIDEO_BG = "#101018"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CASCOS - Control de casco")
        self.root.configure(bg=FONDO)

        self.cola = queue.Queue()
        self.ultimo_frame = None
        self.detener_evt = threading.Event()
        self.hilo = None
        self.model = None
        self.ventana_ajustes = None
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._img = None
        self.corriendo = False

        # Variables de ajustes
        self.v_fuente = tk.StringVar(value=str(pluma.detectar_camara("Web Camera")))
        self.v_usar_ard = tk.BooleanVar(value=True)
        self.v_puerto = tk.StringVar(value=pluma.PUERTO)
        self.v_conf = tk.DoubleVar(value=0.5)
        self.v_seg = tk.IntVar(value=5)
        self.v_espera = tk.DoubleVar(value=1.5)

        # Contadores
        self.n_con = 0
        self.n_sin = 0

        self._construir()
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        # Pantalla completa (F11 alterna, Esc sale)
        self.pantalla_completa = True
        self.root.attributes("-fullscreen", True)
        self.root.bind("<F11>", lambda e: self.alternar_pantalla())
        self.root.bind("<Escape>", lambda e: self.salir_pantalla())

        self.log(f"Dispositivo: {'GPU' if self.device == 0 else 'CPU'}")
        self.log("Cargando modelo...")
        threading.Thread(target=self._cargar_modelo, daemon=True).start()
        self.root.after(30, self._actualizar)

    # ---------------- PANTALLA COMPLETA ----------------
    def alternar_pantalla(self):
        self.pantalla_completa = not self.pantalla_completa
        self.root.attributes("-fullscreen", self.pantalla_completa)

    def salir_pantalla(self):
        self.pantalla_completa = False
        self.root.attributes("-fullscreen", False)

    # ---------------- CONSTRUCCIÓN ----------------
    def _construir(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)  # el cuerpo (video) ocupa el espacio sobrante

        # Encabezado
        encabezado = tk.Frame(self.root, bg=FONDO, padx=20, pady=12)
        encabezado.grid(row=0, column=0, sticky="ew")

        titulos = tk.Frame(encabezado, bg=FONDO)
        titulos.pack(side="left")
        tk.Label(titulos, text="CASCOS", font=("Segoe UI", 26, "bold"), bg=FONDO, fg=TEXTO).pack(anchor="w")
        tk.Label(titulos, text="Control de casco  -  Detección vehicular inteligente",
                 font=("Segoe UI", 10), bg=FONDO, fg=TEXTO_SUAVE).pack(anchor="w")

        derecha = tk.Frame(encabezado, bg=FONDO)
        derecha.pack(side="right", anchor="ne")
        self.lbl_estado = tk.Label(derecha, text="Estado: Cargando modelo", font=("Segoe UI", 11, "bold"),
                                   bg=FONDO, fg="#D97706")
        self.lbl_estado.pack(anchor="e")
        self.lbl_pluma = tk.Label(derecha, text="Pluma: --", font=("Segoe UI", 10),
                                  bg=FONDO, fg=TEXTO_SUAVE)
        self.lbl_pluma.pack(anchor="e", pady=(4, 0))

        # Tarjetas de indicadores
        tarjetas = tk.Frame(self.root, bg=FONDO, padx=20)
        tarjetas.grid(row=1, column=0, sticky="ew")
        self.kpi_con = self._tarjeta(tarjetas, "Con casco hoy", "0", VERDE)
        self.kpi_sin = self._tarjeta(tarjetas, "Sin casco hoy", "0", ROJO)
        self.kpi_estado = self._tarjeta(tarjetas, "Último evento", "-", TEXTO)

        tk.Frame(self.root, bg=BORDE, height=1).grid(row=2, column=0, sticky="ew", padx=20, pady=12)

        # Cuerpo: video (crece con la ventana) + botones
        cuerpo = tk.Frame(self.root, bg=FONDO, padx=20)
        cuerpo.grid(row=3, column=0, sticky="nsew")
        cuerpo.columnconfigure(0, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        marco_video = tk.Frame(cuerpo, bg=BORDE, padx=3, pady=3)
        marco_video.grid(row=0, column=0, sticky="nsew")
        marco_video.columnconfigure(0, weight=1)
        marco_video.rowconfigure(0, weight=1)
        self.video = tk.Canvas(marco_video, bg=VIDEO_BG, highlightthickness=0)
        self.video.grid(row=0, column=0, sticky="nsew")

        lateral = tk.Frame(cuerpo, bg=FONDO, padx=14)
        lateral.grid(row=0, column=1, sticky="ns")
        self.btn_iniciar = self._boton(lateral, "Iniciar", "#D6E8F7", "#1E4E79", self.iniciar, deshabilitado=True)
        self.btn_detener = self._boton(lateral, "Detener", "#F5D5D5", "#8B1A1A", self.detener, deshabilitado=True)
        self._boton(lateral, "Ajustes", "#E2E2E2", TEXTO, self.abrir_ajustes)
        self._boton(lateral, "Cargar video prueba", "#EDE7F6", "#4C1D95", self.cargar_video)
        self._boton(lateral, "Pantalla completa (F11)", GRIS_BTN, TEXTO, self.alternar_pantalla, pequeno=True)
        tk.Frame(lateral, bg=BORDE, height=1).pack(fill="x", pady=10)
        self._boton(lateral, "Limpiar registro", GRIS_BTN, TEXTO, self.limpiar_registro, pequeno=True)
        self._boton(lateral, "Reiniciar contadores", GRIS_BTN, TEXTO, self.reiniciar_contadores, pequeno=True)

        # Registro
        registro_marco = tk.Frame(self.root, bg=FONDO, padx=20)
        registro_marco.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        tk.Label(registro_marco, text="Registro de eventos", font=("Segoe UI", 10, "bold"),
                 bg=FONDO, fg=TEXTO).pack(anchor="w", pady=(0, 2))
        self.registro = scrolledtext.ScrolledText(registro_marco, height=6, state="disabled",
                                                  font=("Consolas", 9), bg=TARJETA, relief="flat",
                                                  highlightbackground=BORDE, highlightthickness=1)
        self.registro.pack(fill="x")

        # Pie de página
        pie = tk.Frame(self.root, bg=FONDO, padx=20, pady=8)
        pie.grid(row=5, column=0, sticky="ew")
        tk.Label(pie, text="Modelo: best.pt  |  Clases: casco, no-casco",
                 font=("Segoe UI", 8), bg=FONDO, fg=TEXTO_SUAVE).pack(side="left")
        tk.Label(pie, text="CASCOS AI (r) 2026 - YOLO11 + OpenCV", font=("Segoe UI", 8),
                 bg=FONDO, fg=TEXTO_SUAVE).pack(side="right")

    def _tarjeta(self, padre, titulo, valor, color):
        marco = tk.Frame(padre, bg=BORDE, padx=1, pady=1)
        marco.pack(side="left", padx=(0, 12), pady=(0, 4))
        interior = tk.Frame(marco, bg=TARJETA, padx=18, pady=8)
        interior.pack()
        tk.Label(interior, text=titulo, font=("Segoe UI", 9), bg=TARJETA, fg=TEXTO_SUAVE).pack()
        lbl = tk.Label(interior, text=valor, font=("Segoe UI", 20, "bold"), bg=TARJETA, fg=color)
        lbl.pack()
        return lbl

    def _boton(self, padre, texto, fondo, color_texto, comando, deshabilitado=False, pequeno=False):
        fuente = ("Segoe UI", 9 if pequeno else 11, "bold" if not pequeno else "normal")
        btn = tk.Button(padre, text=texto, command=comando, bg=fondo, fg=color_texto,
                        activebackground=fondo, activeforeground=color_texto,
                        relief="flat", bd=0, font=fuente, width=22,
                        pady=8 if not pequeno else 4, cursor="hand2")
        btn.pack(pady=(0, 6))
        if deshabilitado:
            btn.configure(state="disabled")
        return btn

    # ---------------- AJUSTES ----------------
    def abrir_ajustes(self):
        if self.ventana_ajustes is not None and self.ventana_ajustes.winfo_exists():
            self.ventana_ajustes.lift()
            return
        v = tk.Toplevel(self.root)
        v.title("Ajustes")
        v.configure(bg=FONDO)
        v.resizable(False, False)
        self.ventana_ajustes = v

        cont = tk.Frame(v, bg=FONDO, padx=20, pady=16)
        cont.pack()

        def etiqueta(texto):
            tk.Label(cont, text=texto, font=("Segoe UI", 10, "bold"), bg=FONDO, fg=TEXTO).pack(anchor="w", pady=(10, 2))

        etiqueta("Fuente de video")
        tk.Entry(cont, textvariable=self.v_fuente, width=42).pack(anchor="w")
        tk.Label(cont, text="0 = webcam   |   rtsp://...   |   http://...", font=("Segoe UI", 8),
                 bg=FONDO, fg=TEXTO_SUAVE).pack(anchor="w")

        etiqueta("Arduino")
        tk.Checkbutton(cont, text="Usar Arduino (pluma)", variable=self.v_usar_ard,
                       bg=FONDO, activebackground=FONDO).pack(anchor="w")
        fila = tk.Frame(cont, bg=FONDO)
        fila.pack(anchor="w")
        tk.Label(fila, text="Puerto:", bg=FONDO).pack(side="left")
        tk.Entry(fila, textvariable=self.v_puerto, width=10).pack(side="left", padx=6)

        etiqueta("Confianza mínima")
        fila_c = tk.Frame(cont, bg=FONDO)
        fila_c.pack(anchor="w")
        lbl_c = tk.Label(fila_c, text=f"{self.v_conf.get():.2f}", width=5, bg=FONDO)
        ttk.Scale(fila_c, from_=0.1, to=0.95, variable=self.v_conf, length=240,
                  command=lambda val: lbl_c.configure(text=f"{float(val):.2f}")).pack(side="left")
        lbl_c.pack(side="left", padx=6)

        etiqueta("Segundos de espera de detección")
        tk.Spinbox(cont, from_=0.5, to=30, increment=0.5, format="%.1f",
                   textvariable=self.v_espera, width=6).pack(anchor="w")
        tk.Label(cont, text="Tiempo que debe mantenerse la detección antes de actuar",
                 font=("Segoe UI", 8), bg=FONDO, fg=TEXTO_SUAVE).pack(anchor="w")

        etiqueta("Segundos con la pluma arriba")
        tk.Spinbox(cont, from_=1, to=60, textvariable=self.v_seg, width=6).pack(anchor="w")

        tk.Button(cont, text="Cerrar", command=v.destroy, bg=GRIS_BTN, fg=TEXTO, relief="flat",
                  font=("Segoe UI", 10), padx=20, pady=4).pack(pady=(16, 0))

    # ---------------- UTILIDADES ----------------
    def log(self, texto):
        hora = time.strftime("%H:%M:%S")
        self.registro.configure(state="normal")
        self.registro.insert("end", f"[{hora}] {texto}\n")
        self.registro.see("end")
        self.registro.configure(state="disabled")

    def cargar_video(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona un video de prueba",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        self.v_fuente.set(ruta)
        self.log(f"Video de prueba: {os.path.basename(ruta)}")
        if self.corriendo:
            self.log("Detén la detección y presiona INICIAR para usar el video.")
        else:
            self.iniciar()

    def limpiar_registro(self):
        self.registro.configure(state="normal")
        self.registro.delete("1.0", "end")
        self.registro.configure(state="disabled")

    def reiniciar_contadores(self):
        self.n_con = 0
        self.n_sin = 0
        self.kpi_con.configure(text="0")
        self.kpi_sin.configure(text="0")

    def set_estado(self, texto, color):
        self.lbl_estado.configure(text=f"Estado: {texto}", fg=color)

    # ---------------- MODELO ----------------
    def _cargar_modelo(self):
        try:
            with open(f"{RUTA_DATASET}\\data.yaml", encoding="utf-8") as f:
                datos = yaml.safe_load(f)
            self.cola.put(("log", f"Clases del dataset: {datos['names']}"))
            model = YOLO(RUTA_MODELO)
            model.fuse()
            self.model = model
            self.cola.put(("log", "Modelo listo."))
            self.cola.put(("listo", None))
        except Exception as e:
            self.cola.put(("log", f"ERROR al cargar el modelo: {e}"))
            self.cola.put(("estado", ("Error de modelo", ROJO)))

    # ---------------- CONTROL ----------------
    def iniciar(self):
        texto = self.v_fuente.get().strip()
        if not texto:
            self.log("Escribe una fuente de video en Ajustes.")
            return
        fuente = int(texto) if texto.isdigit() else texto
        es_archivo = isinstance(fuente, str) and os.path.isfile(fuente)
        self.detener_evt.clear()
        self.corriendo = True
        self.btn_iniciar.configure(state="disabled")
        self.btn_detener.configure(state="normal")
        self.set_estado("Detectando", VERDE)
        self.hilo = threading.Thread(
            target=self._bucle,
            args=(fuente, self.v_conf.get(), max(1, self.v_seg.get()),
                  self.v_usar_ard.get(), self.v_puerto.get().strip(),
                  max(0.5, self.v_espera.get()), es_archivo),
            daemon=True,
        )
        self.hilo.start()

    def detener(self):
        self.detener_evt.set()
        self.btn_detener.configure(state="disabled")

    def cerrar(self):
        self.detener_evt.set()
        self.root.after(300, self.root.destroy)

    # ---------------- HILO DE DETECCIÓN ----------------
    def _estado_frame(self, result):
        detectados = {self.model.names[int(c)] for c in result.boxes.cls}
        if SIN_CASCO in detectados:
            return "SIN_CASCO"
        if CON_CASCO in detectados:
            return "CON_CASCO"
        return None

    def _bucle(self, fuente, conf, segundos_pluma, usar_ard, puerto, espera, es_archivo=False):
        if isinstance(fuente, int):
            cap = cv2.VideoCapture(fuente, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(fuente)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.cola.put(("log", f"No se pudo abrir la cámara: {fuente}"))
            self.cola.put(("fin", None))
            return
        self.cola.put(("log", "Video de prueba abierto." if es_archivo else "Cámara conectada."))
        # Velocidad real para archivos de video (sin esto se procesan lo más rápido posible)
        fps = cap.get(cv2.CAP_PROP_FPS) if es_archivo else 0
        intervalo = 1.0 / fps if fps and fps > 0 else 0

        if usar_ard:
            pluma.conectar(puerto)
            if pluma._arduino is None:
                self.cola.put(("log", "Arduino no conectado: modo simulación."))
                self.cola.put(("pluma", "Simulación"))
            else:
                self.cola.put(("log", f"Arduino conectado en {puerto}."))
                self.cola.put(("pluma", "Conectada"))

        if self.model is None:
            self.cola.put(("log", "El modelo aún no está listo."))
            self.cola.put(("fin", None))
            return

        ventana = []          # (momento, estado) de los últimos "espera" segundos
        estado_actual = None
        pluma_abierta = False
        t_apertura = 0.0

        while not self.detener_evt.is_set():
            inicio_ciclo = time.time()
            ok, frame = cap.read()
            if not ok:
                if es_archivo:
                    self.cola.put(("log", "Fin del video de prueba."))
                else:
                    self.cola.put(("log", "Se perdió la señal de la cámara."))
                break

            result = self.model.predict(frame, conf=conf, imgsz=IMGSZ, device=self.device,
                                        half=(self.device == 0), verbose=False)[0]

            # Confirmación por tiempo: la detección debe mantenerse "espera" segundos
            ahora = time.time()
            ventana.append((ahora, self._estado_frame(result)))
            ventana = [(t, e) for t, e in ventana if ahora - t <= espera]
            confirmado = False
            if ahora - ventana[0][0] >= espera * 0.8:
                estado_frame, veces = Counter(e for _, e in ventana).most_common(1)[0]
                confirmado = veces / len(ventana) >= 0.6
            if confirmado and estado_frame is None and estado_actual is not None:
                # Ya no hay nadie frente a la cámara: se limpia el estado
                estado_actual = None
                self.cola.put(("banner", ("Sin detección", TEXTO_SUAVE)))
            elif confirmado and estado_frame is not None and estado_frame != estado_actual:
                estado_actual = estado_frame
                if estado_actual == "CON_CASCO":
                    self.cola.put(("evento", "CON_CASCO"))
                    self.cola.put(("log", "Motociclista con casco: puede pasar."))
                    if usar_ard and not pluma_abierta:
                        pluma.abrir()
                        pluma_abierta = True
                        t_apertura = time.time()
                        self.cola.put(("pluma", "Arriba"))
                else:
                    self.cola.put(("evento", "SIN_CASCO"))
                    self.cola.put(("log", "ALERTA: persona sin casco."))
                    if usar_ard and pluma_abierta:
                        pluma.cerrar()
                        pluma_abierta = False
                        self.cola.put(("pluma", "Abajo"))

            if pluma_abierta and time.time() - t_apertura > segundos_pluma:
                if usar_ard:
                    pluma.cerrar()
                pluma_abierta = False
                self.cola.put(("pluma", "Abajo"))

            if estado_actual == "SIN_CASCO":
                self.cola.put(("banner", ("PERSONA SIN CASCO", ROJO)))
            elif estado_actual == "CON_CASCO":
                self.cola.put(("banner", ("CON CASCO - PUEDE PASAR", VERDE)))

            for box in result.boxes:
                nombre = self.model.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if nombre == CON_CASCO else (0, 0, 255) if nombre == SIN_CASCO else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{nombre} {float(box.conf):.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.ultimo_frame = frame

            if intervalo:
                espera_frame = intervalo - (time.time() - inicio_ciclo)
                if espera_frame > 0:
                    time.sleep(espera_frame)

        if usar_ard:
            if pluma_abierta:
                pluma.cerrar()
            pluma.desconectar()
        cap.release()
        self.ultimo_frame = None
        self.cola.put(("fin", None))

    # ---------------- ACTUALIZACIÓN (hilo principal) ----------------
    def _actualizar(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "log":
                    self.log(dato)
                elif tipo == "estado":
                    self.set_estado(*dato)
                elif tipo == "listo":
                    self.btn_iniciar.configure(state="normal")
                    self.set_estado("Inactivo", ROJO)
                elif tipo == "evento":
                    if dato == "CON_CASCO":
                        self.n_con += 1
                        self.kpi_con.configure(text=str(self.n_con))
                        self.kpi_estado.configure(text="Con casco", fg=VERDE)
                    else:
                        self.n_sin += 1
                        self.kpi_sin.configure(text=str(self.n_sin))
                        self.kpi_estado.configure(text="Sin casco", fg=ROJO)
                elif tipo == "pluma":
                    self.lbl_pluma.configure(text=f"Pluma: {dato}")
                elif tipo == "banner":
                    texto, color = dato
                    self.kpi_estado.configure(text=texto, fg=color, font=("Segoe UI", 11, "bold"))
                elif tipo == "fin":
                    self.corriendo = False
                    self.btn_iniciar.configure(state="normal")
                    self.btn_detener.configure(state="disabled")
                    self.set_estado("Inactivo", ROJO)
                    self._img = None
                    self.log("Detenido.")
        except queue.Empty:
            pass

        # Dibuja el video escalado al tamaño actual del área, manteniendo proporción
        cw = self.video.winfo_width()
        ch = self.video.winfo_height()
        if cw > 10 and ch > 10:
            self.video.delete("all")
            frame = self.ultimo_frame
            if frame is None:
                self.video.create_text(cw // 2, ch // 2, fill="#A5B4C8", font=("Segoe UI", 14),
                                       justify="center",
                                       text="Presiona INICIAR\n(conecta la cámara en Ajustes)")
            else:
                h, w = frame.shape[:2]
                escala = min(cw / w, ch / h)
                nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
                self._img = ImageTk.PhotoImage(img)
                self.video.create_image(cw // 2, ch // 2, image=self._img)

        self.root.after(30, self._actualizar)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
