from django import forms
from .models import LLMMapping
import json

def get_llm_choices():
    """
    Retrieves all LLMMapping entries from the database and creates a list of
    tuples suitable for a ChoiceField.
    """
    llm_queryset = LLMMapping.objects.all()
    choices = [("", "Select a model (optional)")] + [
        (llm.name, llm.name.replace("_", " ").title()) for llm in llm_queryset
    ]
    return choices

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
    
    # Use the field name "selected_llm" throughout.
    selected_llm = forms.ChoiceField(choices=[], required=True)

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
        help_text="Enter the # of parameters in billions (e.g., 7.0 for 7B)",
        min_value=1
    )

    quantization_level = forms.ChoiceField(
        choices=QUANTIZATION_CHOICES,
        required=True,
        help_text="Select the quantization level."
    )

    context_window = forms.IntegerField(
        required=False,
        help_text="Enter the context window size (e.g., 2048)",
        min_value=1
    )

    cache_bit = forms.IntegerField(
        required=False,
        help_text="Enter the cache bit size (e.g., 16 or 8)",
        min_value=1
    )

    num_attention_heads = forms.IntegerField(
        required=False,
        help_text="Enter the number of attention heads",
        min_value=1
    )

    num_key_value_heads = forms.IntegerField(
        required=False,
        help_text="Enter the number of key-value heads",
        min_value=1
    )

    hidden_size = forms.IntegerField(
        required=False,
        help_text="Enter the hidden size (e.g., 4096)",
        min_value=1
    )

    num_hidden_layers = forms.IntegerField(
        required=False,
        help_text="Enter the number of hidden layers",
        min_value=1
    )
    
    ram = forms.FloatField(
        required=True,
        help_text="Enter your system's RAM in GB (optional if not using the .exe).",
        min_value=1
    )

    gpu_vram = forms.FloatField(
        required=True,
        help_text="Enter your GPU VRAM in GB (optional if not using the .exe).",
        min_value=1
    )

    gpu_bandwidth = forms.FloatField(
        required=False,
        help_text="Enter your GPUs bandwidth",
        min_value=1
    )

    ram_bandwidth = forms.FloatField(
        required=False,
        help_text="Enter your RAMs bandwidth",
        min_value=1
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Update the "selected_llm" field choices using the helper function.
        self.fields["selected_llm"].choices = get_llm_choices()

class SystemInformation(forms.Form):
    system_ram = forms.IntegerField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
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
        help_text="Enter your GPU VRAM in GB (optional if not using the .exe).",
        min_value=1
    )

    context_window = forms.FloatField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'oninput': 'vramOutput.value = this.value'
        }),
        help_text="Enter your wished context window size.",
        min_value=1
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