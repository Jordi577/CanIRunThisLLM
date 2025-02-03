class ModelVRAMCalculator:
    def __init__(self, model_config: dict, parameters: int, quant_level: str, context_window: int, cache_bit: int, cuda_overhead = 0.5):
        # Store the model configuration and inputs
        self.model_config = model_config
        self.parameters = parameters
        self.quant_level = quant_level
        self.context_window = context_window
        self.cache_bit = cache_bit
        self.cuda_overhead = cuda_overhead
        # For simplification we assume lm_head_bit = quant_level
        self.head_bit = self.quant_level
    
    def model_weights(self):
        # Define quantization factors for different precision levels
        bytes_per_weight = {
            "fp32": 4,
            "fp16": 2,  # 16-bit precision (default)
            "q8": 8/8,    # 8-bit quantization (2x smaller)
            "q7": 7/8,   # 7-bit quantization (~2.3x smaller)
            "q6": 6/8,   # 6-bit quantization (~2.7x smaller)
            "q5": 5/8,   # 5-bit quantization (~3.2x smaller)
            "q4": 4/8,   # 4-bit quantization (4x smaller)
            "q3": 3/8,   # 3-bit quantization (~5.3x smaller)
            "q2": 2/8,   # 2-bit quantization (8x smaller)
            "q1": 1/8    # 1-bit quantization (16x smaller)
        }

        # Check if the quantization level is valid
        if self.quant_level not in bytes_per_weight:
            return "Error: Invalid quantization type"
        
        # Calculate VRAM for model weights
        vram = self.parameters * bytes_per_weight[self.quant_level]
        return round(vram, 2)
    
    def kv_cache(self):
        # Grouped Query Attention calculation
        n_gqa = self.model_config["num_attention_heads"] / self.model_config["num_key_value_heads"]
        n_embd_gpa = self.model_config["hidden_size"] / n_gqa
        n_elements = n_embd_gpa * (self.model_config["num_hidden_layers"] * self.context_window)
        
        # Cache size is proportional to number of elements and cache_bit
        cache = 2 * n_elements
        return round(cache * (self.cache_bit / 8) / 1e9, 2)
    
    def activation_vram(self):
        head_dim = self.model_config["hidden_size"] / self.model_config["num_attention_heads"]
        q = self.bytes_per_weight * self.context_window * head_dim * self.model_config["num_attention_heads"]
        k = self.bytes_per_weight * self.context_window * head_dim * self.model_config["num_key_value_heads"]
        v = self.bytes_per_weight * self.context_window * head_dim * self.model_config["num_key_value_heads"]
        # under work
        # softmax_dropout_mask = self.model_config["num_attention_heads"] * self.context_window

    def total_vram(self):
        # Get model weights VRAM
        vram_model_weight = self.model_weights()

        # Get KV cache VRAM
        vram_kv_cache = self.kv_cache()

        # Return the total VRAM needed (sum of model weights and KV cache VRAM)
        total_vram = vram_model_weight + vram_kv_cache
        return round(total_vram + self.cuda_overhead, 2)
    
    def cuda_buffer(self):
        return self.cuda_overhead
    
    def compute_vram_simple(self):
        return self.model_weights()

    def compute_vram_advanced(self):
         # Get model weights VRAM
        vram_model_weight = self.model_weights()

        # Get KV cache VRAM
        vram_kv_cache = self.kv_cache()

        # Return the total VRAM needed (sum of model weights and KV cache VRAM)
        total_vram = vram_model_weight + vram_kv_cache
        return round(total_vram + self.cuda_overhead, 2)