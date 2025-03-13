class ModelVRAMCalculator:
    def __init__(self, model_config: dict, parameters: int, quant_level: str, context_window: int, cache_bit: int, cuda_overhead=0.5):
        self.model_config = model_config
        self.parameters = parameters
        self.quant_level = quant_level
        self.context_window = context_window
        self.cache_bit = cache_bit
        self.cuda_overhead = cuda_overhead
        self.head_bit = self.quant_level # For simplification we assume lm_head_bit = quant_level

    def model_weights(self):
        # Define quantization factors for different precision levels
        bytes_per_weight = {
            "fp32":     4,
            "fp16":     2,
            "q8":       8 / 8,
            "q7":       7 / 8,
            "q6_k":     6.5 / 8,
            "q6":       6 / 8,
            "q5_k_s":   4.8 / 8,
            "q5_k_m":   5.5 / 8,
            "q5":       5 / 8,
            "q4_k_s":   4.3 / 8,
            "q4_k_m":   4.6 / 8,
            "q4":       4 / 8,
            "q3_k_s":   3.4 / 8,
            "q3_k_m":   3.5 / 8,
            "q3_k_l":   3.9 / 8,
            "q3":       3 / 8,
            "q2_k":     2.56 / 8,
            "q2":       2 / 8,
            "q1":       1 / 8,
        }

        if self.quant_level not in bytes_per_weight:
            return "Error: Invalid quantization type"

        vram = self.parameters * bytes_per_weight[self.quant_level]

        return round(vram, 2)

    def kv_cache(self):
        # Grouped Query Attention calculation
        n_gqa = self.model_config["num_attention_heads"] / \
            self.model_config["num_key_value_heads"]
        n_embd_gpa = self.model_config["hidden_size"] / n_gqa
        n_elements = n_embd_gpa * \
            (self.model_config["num_hidden_layers"] * self.context_window)

        # Cache size is proportional to number of elements and cache_bit
        cache = 2 * n_elements

        return round(cache * (self.cache_bit / 8) / 1e9, 2)

    def activation_vram(self):
        head_dim = self.model_config["hidden_size"] / \
            self.model_config["num_attention_heads"]
        q = self.bytes_per_weight * self.context_window * \
            head_dim * self.model_config["num_attention_heads"]
        k = self.bytes_per_weight * self.context_window * \
            head_dim * self.model_config["num_key_value_heads"]
        v = self.bytes_per_weight * self.context_window * \
            head_dim * self.model_config["num_key_value_heads"]
        # Future Work
        # softmax_dropout_mask = self.model_config["num_attention_heads"] * self.context_window

    def cuda_buffer(self):

        return self.cuda_overhead

    def compute_vram_simple(self):

        return self.model_weights()

    def compute_vram_advanced(self):
        vram_model_weight = self.model_weights()
        vram_kv_cache = self.kv_cache()
        total_vram = vram_model_weight + vram_kv_cache

        return round(total_vram + self.cuda_overhead, 2)
