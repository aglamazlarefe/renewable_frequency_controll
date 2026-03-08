import torch
print("CUDA Aktif mi?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Kullanılan Ekran Kartı:", torch.cuda.get_device_name(0))
else:
    print("CUDA desteklenmiyor.")