import time

import serial
from serial.tools import list_ports

BAUDIOS = 9600

# Identificadores USB comunes: Arduino oficial, Arduino (clon), CH340, FTDI
_VIDS_ARDUINO = {0x2341, 0x2A03, 0x1A86, 0x0403}


def detectar_arduino():
    """Busca el puerto COM donde está conectado el Arduino. Devuelve None si no lo encuentra."""
    for p in list_ports.comports():
        texto = f"{p.description or ''} {p.manufacturer or ''}".lower()
        if "arduino" in texto or p.vid in _VIDS_ARDUINO:
            return p.device
    return None


def listar_camaras():
    """Devuelve [(indice, nombre), ...] de todas las cámaras que ve Windows (DirectShow)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        return []


def elegir_camara():
    """Muestra las cámaras disponibles y deja elegir una por consola. Devuelve el índice."""
    camaras = listar_camaras()
    if not camaras:
        print("[CAMARA] No se detectaron cámaras por nombre. Usando índice 0.")
        return 0
    print("\nCámaras disponibles:")
    for i, nombre in camaras:
        print(f"  {i}: {nombre}")
    while True:
        eleccion = input("Elige el número de la cámara a usar: ").strip()
        if eleccion.isdigit() and int(eleccion) in [i for i, _ in camaras]:
            return int(eleccion)
        print("Número no válido, intenta de nuevo.")


def detectar_camaras_con_nombre(nombre="Web Camera"):
    """Devuelve los índices de OpenCV de todas las cámaras cuyo nombre contiene 'nombre'."""
    return [i for i, n in listar_camaras() if nombre.lower() in n.lower()]


def detectar_camara(nombre="Web Camera", max_indices=5):
    """Devuelve el índice de OpenCV de la primera cámara con ese nombre.
    Si no encuentra el nombre o falta pygrabber, usa el primer índice que responda."""
    import cv2
    encontrados = detectar_camaras_con_nombre(nombre)
    if encontrados:
        return encontrados[0]
    for i in range(max_indices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return i
    return None


PUERTO = detectar_arduino() or ""

_arduino = None


def conectar(puerto=None, baudios=BAUDIOS):
    """Abre la conexión serial con el Arduino. Si no se indica puerto, lo detecta solo.
    Si no hay Arduino, funciona en modo simulación."""
    global _arduino
    if not puerto:
        puerto = detectar_arduino()
    if not puerto:
        _arduino = None
        print("[PLUMA] No se encontró ningún Arduino. Modo simulación.")
        return
    try:
        _arduino = serial.Serial(puerto, baudios, timeout=1)
        time.sleep(2)  # El Arduino se reinicia al abrir el puerto
        print(f"[PLUMA] Arduino conectado en {puerto}")
    except serial.SerialException as e:
        _arduino = None
        print(f"[PLUMA] No se pudo conectar al Arduino ({e}). Modo simulación.")


def _enviar(senal):
    if _arduino is not None and _arduino.is_open:
        _arduino.write(senal.encode())
    print(f"[PLUMA] Señal enviada: {senal}")


def abrir():
    """Levanta la pluma."""
    _enviar("A")


def cerrar():
    """Baja la pluma."""
    _enviar("C")


def denegar():
    """Muestra SIN CASCO / ACCESO DENEGADO con LED rojo y buzzer."""
    _enviar("N")


def verificar():
    """Muestra VERIFICANDO en la pantalla LCD."""
    _enviar("V")


def esperar_fin_ciclo(timeout=15):
    """Espera la señal CICLO_TERMINADO que envía el Arduino al terminar un ciclo."""
    if _arduino is None or not _arduino.is_open:
        return True  # En modo simulación no hay nada que esperar
    fin = time.time() + timeout
    while time.time() < fin:
        linea = _arduino.readline().decode(errors="ignore").strip()
        if linea:
            print(f"[ARDUINO] {linea}")
        if linea == "CICLO_TERMINADO":
            return True
    return False


def desconectar():
    global _arduino
    if _arduino is not None and _arduino.is_open:
        _arduino.close()
    _arduino = None


if __name__ == "__main__":
    # Prueba rápida: sube y baja la pluma
    conectar()
    abrir()
    time.sleep(3)
    cerrar()
    desconectar()
