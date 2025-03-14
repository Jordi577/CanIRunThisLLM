from django import forms
from .models import LLMMapping, AppleMSeriesProcessor

# Available configuration modes for model estimation.
CONFIGURATION_MODE_CHOICES = [
    ('simple', 'Simple'),
    ('advanced', 'Advanced'),
]

# Available quantization choices.
QUANTIZATION_CHOICES = [
    ('fp32', 'FP32'),
    ('fp16', 'FP16'),
    ('q8',   'Q8'),
    ('q7',   'Q7'),
    ('q6',   'Q6'),
    ('q6_k', 'Q6_K'),
    ('q5',   'Q5'),
    ('q5_k_m', 'Q5_K_M'),
    ('q5_k_s', 'Q5_K_S'),
    ('q4',   'Q4'),
    ('q4_k_m', 'Q4_K_M'),
    ('q4_k_s', 'Q4_K_S'),
    ('q3',   'Q3'),
    ('q3_k_l', 'Q3_K_L'),
    ('q3_k_m', 'Q3_K_M'),
    ('q3_k_s', 'Q3_K_S'),
    ('q2',   'Q2'),
    ('q2_k', 'Q2_k'),
    ('q1',   'Q1'),
]


def get_llm_choices() -> list[tuple[str, str]]:
    llm_queryset = LLMMapping.objects.all()
    choices = [("", "Select a model (optional)")] + [
        (llm.name, llm.name.replace("_", " ").title()) for llm in llm_queryset
    ]

    return choices


def get_m_series_processors() -> list[tuple[str, str]]:
    m_series = AppleMSeriesProcessor.objects.all()
    choices = [("", "Select a model (optional)")] + [
        (m.name, m.name.replace("_", " ").title()) for m in m_series
    ]

    return choices


class ModelConfigurationForm(forms.Form):
    configuration_mode = forms.ChoiceField(
        choices=CONFIGURATION_MODE_CHOICES,
        required=True,
        initial='simple',
        help_text="Choose the configuration method."
    )

    selected_llm = forms.ModelChoiceField(
        queryset=LLMMapping.objects.order_by('name'),
        required=False,
        empty_label="Select a model"
    )

    predefined_model = forms.ModelChoiceField(
        queryset=LLMMapping.objects.order_by('name'),
        required=False,
        help_text="Or select a predefined model."
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



class WindowsSystemInformationForm(forms.Form):
    windows_gpu_vram = forms.DecimalField(
        required=True,
        help_text="Enter your GPU VRAM in GB (optional if not using the .exe).",
        decimal_places=2,
        min_value=1
    )

    windows_ram = forms.DecimalField(
        required=True,
        help_text="Enter your system RAM in GB (optional if not using the .exe).",
        decimal_places=2,
        min_value=1
    )

    windows_gpu_bandwidth = forms.DecimalField(
        required=False,
        help_text="Enter your GPU's bandwidth.",
        decimal_places=2,
        min_value=1
    )

    windows_ram_bandwidth = forms.DecimalField(
        required=False,
        help_text="Enter your RAM's bandwidth.",
        decimal_places=2,
        min_value=1
    )


class MacOsSystemInformationForm(forms.Form):
    macos_unified_ram = forms.IntegerField(
        required=True,
        help_text="Enter your system's RAM in GB (used as unified memory).",
        min_value=1
    )

    m_series_processor = forms.ChoiceField(choices=[], required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["m_series_processor"].choices = get_m_series_processors()


class SystemInformationForm(forms.Form):
    system_ram = forms.DecimalField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'oninput': 'ramOutput.value = this.value'
        }),
        help_text="Enter your system's RAM in GB.",
        min_value=1,
        decimal_places=2,
    )

    system_vram = forms.DecimalField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'min': '1',
            'max': '8192',
            'step': '1',
            'oninput': 'vramOutput.value = this.value'
        }),
        help_text="Enter your GPU VRAM in GB.",
        min_value=1,
        decimal_places=2
    )

    context_window = forms.DecimalField(
        required=True,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'oninput': 'vramOutput.value = this.value'
        }),
        help_text="Enter your desired context window size.",
        min_value=1,
        decimal_places=2
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
