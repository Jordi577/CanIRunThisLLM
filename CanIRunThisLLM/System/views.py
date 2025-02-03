from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .static.vram_calc import ModelVRAMCalculator
from .static.canirunit import CanIRunIt
import json

from .forms import VRAMCalculationForm, LLM_CHOICES_MAPPING


@api_view(['POST'])
def upload_system_info(request):
    """
    Receives system info from the client (the .exe) and stores it in the session.
    Optionally, you can store it in a cookie if desired.
    """
    # Store the data in the session
    request.session["system_information"] = request.data
    print("System information stored in session:",
          request.session.get("system_information"))

    response = Response(
        {"message": "Data received successfully!", "reload": True},
        status=200
    )

    response.set_cookie("system_information",
                        json.dumps(request.data), max_age=3600)

    return response


def home(request):
    form = VRAMCalculationForm(request.POST or None)
    can_run_result = None

    if request.method == 'POST' and form.is_valid():
        parameters_model = form.cleaned_data['parameters_model']
        quantization_level = form.cleaned_data['quantization_level']
        context_window = form.cleaned_data['context_window']
        cache_bit = form.cleaned_data['cache_bit']
        num_attention_heads = form.cleaned_data['num_attention_heads']
        num_key_value_heads = form.cleaned_data['num_key_value_heads']
        hidden_size = form.cleaned_data['hidden_size']
        num_hidden_layers = form.cleaned_data['num_hidden_layers']
        configuration_mode = form.cleaned_data.get(
            "configuration_mode", "simple")
        print(parameters_model)

        # --- VRAM Calculation ---
        vram_calculator = ModelVRAMCalculator(
            model_config={
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "hidden_size": hidden_size,
                "num_hidden_layers": num_hidden_layers,
            },
            parameters=parameters_model,
            quant_level=quantization_level,
            context_window=context_window,
            cache_bit=cache_bit
        )

        manual_ram = form.cleaned_data.get('ram')
        manual_vram = form.cleaned_data.get('gpu_vram')

        if configuration_mode == "simple":
            total_vram = vram_calculator.compute_vram_simple()
        elif configuration_mode == "advanced":
            total_vram = vram_calculator.compute_vram_advanced()
        else:
            total_vram = vram_calculator.compute_vram_intermediate()

        model_weight_vram = vram_calculator.model_weights()
        kv_cache_vram = vram_calculator.kv_cache()
        cuda_buffer_vram = vram_calculator.cuda_buffer()

        # --- "Can I Run It?" Check ---
        if "run_check" in request.POST:
            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=manual_vram if manual_vram is not None else 0,
                system_ram=manual_ram if manual_ram is not None else 0
            )
            can_run_result = checker.decide()
            print(can_run_result)

        # Reinitialize the form with submitted values
        form = VRAMCalculationForm(initial=form.cleaned_data)

        return render(request, 'System/home.html', {
            'form': form,
            'vram': total_vram,
            'mode_weight': model_weight_vram,
            'kv_cache': kv_cache_vram,
            'cuda_buffer': cuda_buffer_vram,
            'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
            'can_run_result': can_run_result,
            'ram': manual_ram,
            'gpu_vram': manual_vram
        })

    initial = {
        'parameters_model': 685,
        'quantization_level': 'fp16',
        'context_window': 8192,
        'cache_bit': 16,
        'num_attention_heads': 128,
        'num_key_value_heads': 128,
        'hidden_size': 7168,
        'num_hidden_layers': 61,
        'selected_llm': 'DeepSeek-R1',
        'ram': 16,
        'gpu_vram': 8
    }

    form = VRAMCalculationForm(initial=initial)

    return render(request, 'System/home.html', {
        'form': form,
        'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
    })
