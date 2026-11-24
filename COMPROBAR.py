import torch

print("¿CUDA está disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de la GPU conectada:", torch.cuda.get_device_name(0))
