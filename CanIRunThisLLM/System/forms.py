from django import forms
import json

def read_llm_choices_mapping(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

LLM_CHOICES_MAPPING = read_llm_choices_mapping("System/static/llm_map.json")

LLM_CHOICES = [("", "Select a model (optional)")] + [
    (key, key.replace("_", " ").title()) for key in LLM_CHOICES_MAPPING.keys()
]

CONFIGURATION_MODE_CHOICES = [
    ('simple', 'Simple'),
    ('advanced', 'Advanced'),
]

QUANTIZATION_CHOICES = [
    ('fp32', 'FP32'),
    ('fp16', 'FP16'),
    ('q8', 'Q8'),
    ('q7', 'Q7'),
    ('q6', 'Q6'),
    ('q5', 'Q5'),
    ('q4', 'Q4'),
    ('q3', 'Q3'),
    ('q2', 'Q2'),
    ('q1', 'Q1'),
]

class VRAMCalculationForm(forms.Form):
    configuration_mode = forms.ChoiceField(
        choices=CONFIGURATION_MODE_CHOICES,
        required=True,
        initial='simple',
        help_text="Choose the configuration method."
    )
    
    selected_llm = forms.ChoiceField(
        choices=LLM_CHOICES,
        required=False,
        help_text="Select a predefined model to prepopulate the fields."
    )

    predefined_model_custom = forms.CharField(
        required=False,
        help_text="Or enter a predefined model manually."
    )
    
    huggingface_model_path = forms.CharField(
        required=False,
        help_text="Enter a Hugging Face model path (e.g., 'meta-llama/Llama-2-7b-chat-hf')."
    )

    parameters_model = forms.FloatField(
        required=True,
        help_text="Enter the # of parameters in billions (e.g., 7.0 for 7B)"
    )

    quantization_level = forms.ChoiceField(
        choices=QUANTIZATION_CHOICES,
        required=True,
        help_text="Select the quantization level."
    )

    context_window = forms.IntegerField(
        required=False,
        help_text="Enter the context window size (e.g., 2048)"
    )

    cache_bit = forms.IntegerField(
        required=False,
        help_text="Enter the cache bit size (e.g., 16 or 8)"
    )

    num_attention_heads = forms.IntegerField(
        required=False,
        help_text="Enter the number of attention heads"
    )

    num_key_value_heads = forms.IntegerField(
        required=False,
        help_text="Enter the number of key-value heads"
    )

    hidden_size = forms.IntegerField(
        required=False,
        help_text="Enter the hidden size (e.g., 4096)"
    )

    num_hidden_layers = forms.IntegerField(
        required=False,
        help_text="Enter the number of hidden layers"
    )
    
    ram = forms.FloatField(
        required=True,
        help_text="Enter your system's RAM in GB (optional if not using the .exe)."
    )

    gpu_vram = forms.FloatField(
        required=True,
        help_text="Enter your GPU VRAM in GB (optional if not using the .exe)."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        selected = self.initial.get("selected_llm") or self.data.get("selected_llm")
        if selected and selected in LLM_CHOICES_MAPPING:
            config = LLM_CHOICES_MAPPING[selected]
            self.fields["parameters_model"].initial = config["parameters"] / 1_000_000_000
            self.fields["quantization_level"].initial = config["quant_level"]
            self.fields["context_window"].initial = config["context_window"]
            self.fields["cache_bit"].initial = config["cache_bit"]
            self.fields["num_attention_heads"].initial = config["model_config"]["num_attention_heads"]
            self.fields["num_key_value_heads"].initial = config["model_config"]["num_key_value_heads"]
            self.fields["hidden_size"].initial = config["model_config"]["hidden_size"]
            self.fields["num_hidden_layers"].initial = config["model_config"]["num_hidden_layers"]

class SystemInformation(forms.Form):
    system_ram = forms.IntegerField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'min': '1',
            'max': '8192',  
            'step': '1',
            'oninput': 'ramOutput.value = this.value' 
        }),
        help_text="Enter your system's RAM in GB (optional if not using the .exe)."
    )

    system_vram = forms.FloatField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'min': '1',
            'max': '8192',   
            'step': '1',
            'oninput': 'vramOutput.value = this.value'
        }),
        help_text="Enter your GPU VRAM in GB (optional if not using the .exe)."
    )

    context_window = forms.FloatField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'min': '1',
            'max': '8192',   
            'step': '1',
            'oninput': 'vramOutput.value = this.value'
        }),
        help_text="Enter your wished context windows."
    )

class HuggingfaceModelForm(forms.Form):
    predefined_model_custom = forms.CharField(
        required=False,
        help_text="Enter a predefined model manually."
    )

    huggingface_model_path = forms.CharField(
        required=False,
        help_text="Enter a Hugging Face model path (e.g., 'meta-llama/Llama-2-7b-chat-hf')."
    )
