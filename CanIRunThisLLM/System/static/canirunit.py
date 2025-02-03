class CanIRunIt:
    def __init__(self, required_vram: float, system_vram: float, system_ram: float):
        self.required_vram = required_vram
        self.system_vram = system_vram
        self.system_ram = system_ram
        self.return_types = [1, 2, 3]

    def decide(self) -> str:
        if self.required_vram <= self.system_vram:
            return self.return_types[0]
        elif self.required_vram <= (self.system_vram + self.system_ram):
            return self.return_types[1]
        else:
            return self.return_types[2]

