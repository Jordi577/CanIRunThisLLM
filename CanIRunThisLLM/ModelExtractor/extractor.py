################################################################################################
########### Date: 04.02.2025, 18:42 | Author: Jordi Buscaglia | Location: Hamburg ##############
################################################################################################

from huggingface_hub import hf_hub_download
import json
import os

class ModelExtractor:
    def __init__(self, json_file=None, url=None, hf_token=None):
        self.json_file = json_file      
        self.url = url                  
        self.hf_token = hf_token       
        self.data = None                
        self.all_text = None            
        self.config_available = True
        self.safe_tensor_available = True

    def download_model_config(self, model_name=None):
        repo_id = self.url.split("huggingface.co/")[-1].strip("/")
        model_name = repo_id.split("/")[-1]

        try:
            config_file = hf_hub_download(
                repo_id=repo_id, 
                filename="config.json", 
                local_dir=os.path.join("CanIRunThisLLM", "modelfiles", model_name)
            )

        except:
            print("No config.json available")
            self.config_available = False
            config_file = None 

        try:
            tensor_file = hf_hub_download(
                repo_id=repo_id, 
                filename="model.safetensors.index.json", 
                local_dir=os.path.join("CanIRunThisLLM", "modelfiles", "safetensor", model_name)
            )

        except:
            print("No model.safetensors.index.json available")
            self.safe_tensor_available = False
            tensor_file = None 

        return config_file, tensor_file

    def build_final_config(self, config_path, tensor_file_path, model_name):
        config_data = {}
        tensor_data = {}

        if config_path:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = json.load(file)

        if tensor_file_path:
            with open(tensor_file_path, 'r', encoding='utf-8') as file:
                tensor_data = json.load(file)
        
        def check_value(key, source):
            value = source.get(key)
            return value if value is not None else f"{key} is missing"

        model_config = {
            "num_attention_heads": check_value("num_attention_heads", config_data),
            "num_key_value_heads": check_value("num_key_value_heads", config_data) if "num_key_value_heads" in config_data else check_value("num_attention_heads", config_data),
            "hidden_size": check_value("hidden_size", config_data),
            "num_hidden_layers": check_value("num_hidden_layers", config_data)
        }

        model_information = {
            "name": model_name if self.config_available else "Model name is missing",
            "model_config": model_config if self.config_available else "Model configuration is missing",
            "parameters": check_value("total_size", tensor_data.get("metadata", {})) if self.safe_tensor_available else "Tensor metadata is missing",
            "quant_level": "fp16" if self.config_available else "Quantization level is missing",
            "context_window": 8192 if self.config_available else "Context window information is missing",
            "cache_bit": 16 if self.config_available else "Cache bit information is missing",
            "cuda_overhead": 2 if self.config_available else "CUDA overhead information is missing",
            "config_available": self.config_available,
            "safe_tensor_available": self.safe_tensor_available,
        }
        
        return model_information

if __name__ == '__main__':
    model_path = "asif00/bangla-llama-1B-gguf-16bit"
    website_url = "https://huggingface.co/" + model_path
    model_name = model_path.split('/')[-1]
    extractor = ModelExtractor(url=website_url)
    config_file_path, tensor_file_path = extractor.download_model_config()
    final_config = extractor.build_final_config(config_file_path, tensor_file_path, model_name)
    print(json.dumps(final_config, indent=4))