from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from rest_framework.decorators import api_view
from django.urls import reverse
from django.template.loader import render_to_string
from .forms import VRAMCalculationForm, HuggingfaceModelForm, SystemInformation
from VRAMCalculator.vram_calc import ModelVRAMCalculator
from VRAMCalculator.canirunit import CanIRunIt
from ModelExtractor.extractor import ModelExtractor
from EstimateTokenPerSecond.estimate_token_per_second import EstimateTokenPerSecond
from .models import LLMMapping
import json

def get_llm_choices_mapping():
    mapping = {}
    for entry in LLMMapping.objects.all():
        mapping[entry.name] = {
            "model_config": entry.model_config,
            "parameters": entry.parameters,
            "quant_level": entry.quant_level,
            "context_window": entry.context_window,
            "cache_bit": entry.cache_bit,
            "cuda_overhead": entry.cuda_overhead,
        }
    return mapping


@api_view(['POST'])
def upload_system_info(request):
    request.session["system_information"] = request.data
    request.session["data_send"] = True
    request.session.modified = True

    return HttpResponseRedirect(reverse('home'))


def update_table_view(request):
    try:
        system_ram = float(request.GET.get('system_ram', 0))
        system_vram = float(request.GET.get('system_vram', 0))
        system_context_window = float(request.GET.get('system_context_window', 0))
    except ValueError:
        system_ram = system_vram = system_context_window = 0

    
    print(f"Update table view parameters: ram={system_ram}, vram={system_vram}, context={system_context_window}")

    tks = None
    configuration_mode = request.session.get("configuration_mode", "simple")
    columns = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}

    result = []
    # Instead of iterating over a JSON mapping, query the database.
    for mapping in LLMMapping.objects.all():
        row_result = {"row": mapping.name, "values": []}
        for quant in columns:
            # Use the model's fields directly.
            vram_calculator = ModelVRAMCalculator(
                model_config=mapping.model_config,
                parameters=mapping.parameters / 1e9,
                quant_level=quant,
                context_window=system_context_window,
                cache_bit=mapping.cache_bit
            )

            # Here we use the advanced computation; you could also add logic
            # to choose between compute_vram_simple() or compute_vram_intermediate() if needed.
            total_vram = vram_calculator.compute_vram_advanced()
            checker = CanIRunIt(required_vram=total_vram,
                                system_vram=system_vram,
                                system_ram=system_ram)
            can_run_result = checker.decide()

            if request.session.get("gpu_bandwidth") and request.session.get("ram_bandwidth"):
                estimate_token_per_second = EstimateTokenPerSecond(
                            request.session.get("gpu_bandwidth"), 
                            request.session.get("ram_bandwidth"), 
                            system_vram, system_ram, 
                            quant, 
                            total_vram
                            )        
            
                tks, _ = estimate_token_per_second.calculate_token_per_second()

            row_result["values"].append((colors_map[can_run_result], round(total_vram, 1), tks))
        result.append(row_result)

    context = {"columns": columns, "chart_data": result}
    html = render_to_string("System/partials/table.html", context)
    return HttpResponse(html)


def stop_chart_view(request):
    # Early return if the home button in stop_chart.html was clicked
    if "return_home" in request.POST:
        return HttpResponseRedirect(reverse("home"))

    # Retrieve session parameters
    config_mode = request.session.get("configuration_mode", "simple")
    system_vram = request.session.get("vram", 16)
    system_ram = request.session.get("ram", 8)
    context_window = request.session.get("context_window", 8192)
    gpu_bandwidth = request.session.get("gpu_bandwidth", None)
    ram_bandwidth = request.session.get("ram_bandwidth", None)

    # Prepare initial form data
    initial_data = {
        "system_ram": system_ram,
        "system_vram": system_vram,
        "context_window": context_window,
    }
    system_form = SystemInformation(request.POST or None, initial=initial_data)

    # Define quantization levels and a color mapping for the results
    quant_levels = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}
    chart_data = []

    # Loop through all LLM mappings
    for mapping in LLMMapping.objects.all():
        row_data = {"row": mapping.name, "values": []}

        for quant_level in quant_levels:
            # Instantiate the VRAM calculator for the current mapping
            vram_calculator = ModelVRAMCalculator(
                model_config=mapping.model_config,
                parameters=mapping.parameters / 1e9,
                quant_level=quant_level,
                context_window=mapping.context_window,
                cache_bit=mapping.cache_bit
            )

            # Map configuration mode to the corresponding VRAM computation method
            if config_mode == "simple": total_vram = vram_calculator.compute_vram_simple()
            elif config_mode == "advanced": total_vram = vram_calculator.compute_vram_advanced()
            else: total_vram = vram_calculator.compute_vram_intermediate()

            # Determine if the system can run the model with the calculated VRAM
            can_run = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            ).decide()

            # Estimate tokens per second if bandwidth information is provided
            tks = None
            if gpu_bandwidth and ram_bandwidth:
                tks, _ = EstimateTokenPerSecond(
                    gpu_bandwidth,
                    ram_bandwidth,
                    system_vram,
                    system_ram,
                    quant_level,
                    total_vram
                ).calculate_token_per_second()

            # Append a tuple with the color, total VRAM (rounded), and token estimate
            row_data["values"].append((colors_map[can_run], round(total_vram, 2), tks))

        chart_data.append(row_data)

    # Prepare the context and render the template
    context = {
        "columns": quant_levels,
        "chart_data": chart_data,
        "ram": system_ram,
        "vram": system_vram,
        "system_form": system_form
    }
    return render(request, "System/stop_chart.html", context)


def home(request):
    # Retrieve session values
    ram = request.session.get("ram", 16)
    gpu_vram = request.session.get("vram", 8)
    gpu_bandwidth = request.session.get("gpu_bandwidth")
    ram_bandwidth = request.session.get("ram_bandwidth")

    # Define the initial data for the form and save it in session
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
        'ram': ram,
        'gpu_vram': gpu_vram,
        'gpu_bandwidth': gpu_bandwidth,
        'ram_bandwidth': ram_bandwidth
    }
    request.session["initial"] = initial

    # Instantiate the form and prepare the context
    form = VRAMCalculationForm(request.POST or None, initial=initial)
    context = {
        'form': form,
        'llm_choices_json': json.dumps(get_llm_choices_mapping()),
    }

    # Variables for token estimation
    tks = None
    offload_ratio = None

    if request.method == 'POST' and form.is_valid():
        # Use a separate variable for cleaned data.
        data = form.cleaned_data
        parameters_model = data['parameters_model']
        quantization_level = data['quantization_level']
        context_window = data['context_window']
        cache_bit = data['cache_bit']
        num_attention_heads = data['num_attention_heads']
        num_key_value_heads = data['num_key_value_heads']
        hidden_size = data['hidden_size']
        num_hidden_layers = data['num_hidden_layers']
        configuration_mode = data.get("configuration_mode", "simple")
        gpu_bandwidth = data["gpu_bandwidth"]
        ram_bandwidth = data["ram_bandwidth"]
        manual_ram = data['ram']
        manual_vram = data['gpu_vram']

        # Save user input to the session.
        session_data = {
            "gpu_bandwidth": gpu_bandwidth,
            "ram_bandwidth": ram_bandwidth,
            "vram": manual_vram if manual_vram is not None else 16,
            "ram": manual_ram if manual_ram is not None else 8,
            "configuration_mode": configuration_mode,
        }
        for key, value in session_data.items():
            request.session[key] = value

        # Instantiate the VRAM calculator.
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

        # Compute total VRAM based on configuration mode.
        if configuration_mode == "simple":
            total_vram = vram_calculator.compute_vram_simple()
        elif configuration_mode == "advanced":
            total_vram = vram_calculator.compute_vram_advanced()
        else:
            total_vram = vram_calculator.compute_vram_intermediate()

        # Compute additional VRAM metrics.
        model_weight_vram = vram_calculator.model_weights()
        kv_cache_vram = vram_calculator.kv_cache()
        cuda_buffer_vram = vram_calculator.cuda_buffer()

        can_run_result = None

        if "run_check" in request.POST:
            if gpu_bandwidth and ram_bandwidth:
                estimator = EstimateTokenPerSecond(
                    gpu_bandwidth,
                    ram_bandwidth,
                    manual_vram,
                    manual_ram,
                    quantization_level,
                    total_vram
                )
                tks, offload_ratio = estimator.calculate_token_per_second()

            # (Re)calculate offload_ratio regardless of the token estimator.
            offload_ratio = round(((total_vram - manual_vram) / total_vram * 100), 2)

            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=manual_vram if manual_vram is not None else 0,
                system_ram=manual_ram if manual_ram is not None else 0
            )
            can_run_result = checker.decide()

        elif "stop_light_chart" in request.POST:
            return HttpResponseRedirect(reverse("stop_chart"))

        # Update context with new form and computed values.
        context["form"] = VRAMCalculationForm(initial=data)
        context.update({
            'vram': total_vram,
            'mode_weight': model_weight_vram,
            'kv_cache': kv_cache_vram,
            'cuda_buffer': cuda_buffer_vram,
            'can_run_result': can_run_result,
            'ram': manual_ram,
            'gpu_vram': manual_vram,
            'tks': tks,
            'offload_ratio': offload_ratio,
        })

        return render(request, 'System/home.html', context)

    # For GET or invalid POST, render the form with initial data.
    context["form"] = VRAMCalculationForm(initial=initial)
    return render(request, 'System/home.html', context)