import torch
import psutil

class DeviceManager:
    def __init__(self):
        self.device = self.detect_device()
        self.memory = self.get_memory()

    def detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_memory(self):
        if self.device == "cuda":
            return torch.cuda.get_device_properties(0).total_memory
        else:
            return psutil.virtual_memory().available

    def summary(self):
        return {
            "device": self.device,
            "memory_bytes": self.memory
        }