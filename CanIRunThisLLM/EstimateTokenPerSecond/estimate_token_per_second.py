import math

class EstimateTokenPerSecond:
    """
    Estimate tokens processed per second based on the model's memory requirements
    and available bandwidth of both the GPU and RAM.
    """
    def __init__(
        self,
        gpu_bandwidth: float,
        ram_bandwidth: float,
        gpu_vram: float,
        ram: float,
        quantization_level: str,
        model_memory_print: float,
    ):
        """
        Initialize the estimator.

        Parameters:
            gpu_bandwidth (float): GPU bandwidth (Gb/s).
            ram_bandwidth (float): RAM bandwidth (Gb/s)
            gpu_vram (float): Available GPU VRAM (Gb).
            ram (float): Available system RAM (Gb).
            quantization_level (str): Quantization level ('fp32', 'fp16', 'q8', etc.).
            model_memory_print (float): Amount of the models memory footprint. 
        """
        self.gpu_bandwidth = gpu_bandwidth
        self.ram_bandwidth = ram_bandwidth
        self.gpu_vram = gpu_vram
        self.ram = ram
        self.quantization_level = quantization_level
        self.model_memory_print = model_memory_print

    # Calculate the speed penalty if the model has to be partially loaded into the RAM
    def calc_offload_ratio(self) -> float:
        offload_ratio = round(((self.model_memory_print-self.gpu_vram) / self.model_memory_print * 100), 2)
        return offload_ratio

    # Calculate tokens per second when the model is fully loaded on the GPU.
    def token_per_second(self) -> tuple[float, float]:
        offload_ratio = 100
        tks =  round((self.gpu_bandwidth / self.model_memory_print), 2)
        return tks, offload_ratio
    
    # Calculate tokens per second when the model is partially offloaded.
    def partial_offload_token_per_second(self, base_token_ps: float) -> tuple[float, float]:
        offload_ratio = self.calc_offload_ratio()
        off_tks = round(base_token_ps * (0.052 * math.exp(4.55 * (100 - offload_ratio) / 100) + 1.06), 2)
        return off_tks, offload_ratio
    
    # Calculate tokens per second when the model is fully offloaded to the RAM.
    def full_offload_token_per_second(self) -> tuple[float, float]:
        offload_ratio = 100
        tks = round((self.ram_bandwidth / self.model_memory_print) * 0.9, 2)
        return tks, offload_ratio

    # Estimate tokens per second based on where the model can be loaded.
    def calculate_token_per_second(self) -> tuple[float, float]:
        # Model can be fully loaded into the VRAM
        if self.model_memory_print < self.gpu_vram:
            return self.token_per_second()
        
        # Model has to be offloaded into the RAM
        elif self.model_memory_print < (self.gpu_vram + self.ram):
            base_tks, _ = self.full_offload_token_per_second()  #
            return self.partial_offload_token_per_second(base_tks)
        
        # Model is fully offloaded into the RAM
        else:
            return self.full_offload_token_per_second()

