

from ultralytics import YOLO


def main():
    # Cargar modelo YOLO26 Nano
    model = YOLO("yolo26n.pt")

    # Entrenar detector de cascos
    model.train(
        data=r"C:\Users\PC\Documents\DATASETcasco\Ingelam.v1i.yolo26\data.yaml",
        epochs=30,
        imgsz=416,
        device="0",
        batch=8,
        workers=0,

        # Guardar dentro del proyecto CASCOS
        project=r"C:\Users\PC\PycharmProjects\CASCOS\runs\detect\entrenamientos",
        name="detector_con_casco_ingelam_rapido",
        exist_ok=True
    )

    print("Entrenamiento finalizado correctamente")


if __name__ == "__main__":
    main()

