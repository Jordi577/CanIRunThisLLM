from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from rest_framework.decorators import api_view
from django.urls import reverse
from django.template.loader import render_to_string
from .forms import (
    ModelConfigurationForm,
    WindowsSystemInformationForm,
    MacOsSystemInformationForm,
    SystemInformationForm
)
from VRAMCalculator.vram_calc import ModelVRAMCalculator
from VRAMCalculator.canirunit import CanIRunIt
from ModelExtractor.extractor import ModelExtractor
from EstimateTokenPerSecond.estimate_token_per_second import EstimateTokenPerSecond
from .models import LLMMapping, AppleMSeriesProcessor
import json


def get_llm_choices_mapping():
    """
    Build a dictionary mapping LLM names to their configuration details.
    """
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
    # Save posted system information into the session.
    print(request.data)
    request.session["system_information"] = request.data
    request.session["data_send"] = True
    return HttpResponseRedirect(reverse('home'))


def update_table_view(request):
    try:
        system_ram = float(request.GET.get('system_ram', 0))
        system_vram_val = request.GET.get('system_vram', None)
        if system_vram_val in [None, ""]:
            # If system_vram is not provided, use the unified RAM from session (for macOS).
            system_vram = float(request.session.get("unified_ram", 24))
        else:
            system_vram = float(system_vram_val)
        system_context_window = float(request.GET.get('system_context_window', 0))
    except ValueError:
        system_ram = system_vram = system_context_window = 0

    os_choice = request.session.get("os_choice", "win")

    # Check if a processor value is sent.
    m_processor = request.GET.get('m_processor', None)
    if m_processor:
        request.session["m_processor"] = m_processor
        try:
            apple_processor = AppleMSeriesProcessor.objects.get(name=m_processor)
            # Update both keys used in token estimation.
            request.session["gpu_bandwidth"] = apple_processor.bandwidth
            request.session["ram_bandwidth"] = apple_processor.bandwidth
        except AppleMSeriesProcessor.DoesNotExist:
            pass


    print(f"Update table view parameters: ram={system_ram}, vram={system_vram}, context={system_context_window}, m_processor={m_processor}")

    tks = None
    configuration_mode = request.session.get("configuration_mode", "simple")
    columns = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}

    result = []
    for mapping in LLMMapping.objects.all():
        row_result = {"row": mapping.name, "values": []}
        for quant in columns:
            vram_calculator = ModelVRAMCalculator(
                model_config=mapping.model_config,
                parameters=mapping.parameters / 1e9,
                quant_level=quant,
                context_window=system_context_window,
                cache_bit=mapping.cache_bit
            )
            total_vram = vram_calculator.compute_vram_advanced()
            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            )
            can_run_result = checker.decide()

            if request.session.get("gpu_bandwidth") and request.session.get("ram_bandwidth"):
                estimator = EstimateTokenPerSecond(
                    request.session.get("gpu_bandwidth"),
                    request.session.get("ram_bandwidth"),
                    system_vram, system_ram,
                    quant,
                    total_vram
                )
                tks, _ = estimator.calculate_token_per_second()

            row_result["values"].append((colors_map[can_run_result], round(total_vram, 1), tks))
        result.append(row_result)

    context = {"columns": columns, "chart_data": result, "os_choice": os_choice}
    print(os_choice)
    html = render_to_string("System/partials/table.html", context)
    return HttpResponse(html)


def stop_chart_view(request):
    if "return_home" in request.POST:
        return HttpResponseRedirect(reverse("home"))
    
    elif "mac" in request.POST:
        # Get unified RAM from session.
        unified_ram = request.session.get("unified_ram", 24)
        if unified_ram is None:
            unified_ram = 24
        # Read the processor name from the posted data.
        m_processor = request.POST.get("m_series_processor", "M1")
        try:
            apple_processor = AppleMSeriesProcessor.objects.get(name=m_processor)
            mac_bandwidth = apple_processor.bandwidth
        except AppleMSeriesProcessor.DoesNotExist:
            mac_bandwidth = None
       
        request.session["unified_ram"] = unified_ram
        system_ram = unified_ram
        system_vram = 0
        gpu_bandwidth = mac_bandwidth
        ram_bandwidth = mac_bandwidth  
        os_choice = 'mac'

        print("\n----------------------------")
        print("Os: MACOS")
        print(f"Processor: {m_processor}")
        print(f"Unified RAM (as VRAM): {system_vram}")
        print(f"Bandwidth: {gpu_bandwidth}")
        print("----------------------------\n")
    
    else:
        # Windows branch (unchanged)
        system_vram = request.session.get("system_vram", 16)
        system_ram = request.session.get("system_ram", 8)
        gpu_bandwidth = request.session.get("gpu_bandwidth", None)
        ram_bandwidth = request.session.get("ram_bandwidth", None)
        os_choice = 'win'

        print("\n----------------------------")
        print("Os: Windows")
        print(f"RAM: {system_ram}")
        print(f"VRAM: {system_vram}")
        print(f"GPU Bandwidth: {gpu_bandwidth}")
        print(f"RAM Bandwidth: {ram_bandwidth}")
        print("----------------------------\n")

    if system_vram is None:
        system_vram = 16

    request.session["os_choice"] = os_choice
    config_mode = request.session.get("configuration_mode", "simple")
    context_window = request.session.get("context_window", 8192)

    initial_data = {
        "system_ram": system_ram,
        "system_vram": system_vram,
        "context_window": context_window,
    }

    print(initial_data)

    if 'system_ram' in request.POST:
        system_form = SystemInformationForm(request.POST, initial=initial_data)
    else:
        system_form = SystemInformationForm(initial=initial_data)

    quant_levels = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}
    chart_data = []

    for mapping in LLMMapping.objects.all():
        row_data = {"row": mapping.name, "values": []}
        for quant_level in quant_levels:
            vram_calculator = ModelVRAMCalculator(
                model_config=mapping.model_config,
                parameters=mapping.parameters / 1e9,
                quant_level=quant_level,
                context_window=mapping.context_window,
                cache_bit=mapping.cache_bit
            )
            if config_mode == "simple":
                total_vram = vram_calculator.compute_vram_simple()
            elif config_mode == "advanced":
                total_vram = vram_calculator.compute_vram_advanced()
            else:
                total_vram = vram_calculator.compute_vram_intermediate()

            can_run = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            ).decide()
            tks = None

            if gpu_bandwidth and ram_bandwidth and total_vram != 0:
                tks, _ = EstimateTokenPerSecond(
                    gpu_bandwidth,
                    ram_bandwidth,
                    system_vram,
                    system_ram,
                    quant_level,
                    total_vram
                ).calculate_token_per_second()

            row_data["values"].append((colors_map[can_run], round(total_vram, 2), tks))
        chart_data.append(row_data)

    context = {
        "columns": quant_levels,
        "chart_data": chart_data,
        "ram": system_ram,
        "vram": system_vram,
        "system_form": system_form,
        "os_choice": os_choice
    }

    return render(request, "System/stop_chart.html", context)




def home(request):
    # Retrieve OS-specific session values (if any)
    # windows_ram = request.session.get("ram", 16)
    # windows_vram = request.session.get("vram", 8)
    # windows_gpu_bandwidth = request.session.get("gpu_bandwidth")
    # windows_ram_bandwidth = request.session.get("ram_bandwidth")
    
    unified_ram = request.session.get("unified_ram", 8)  
    m_processor = request.session.get("m_processor", "M1")  

    # Instantiate the model configuration form with initial values.
    model_config_initial = {
        'parameters_model': 685,
        'quantization_level': 'fp16',
        'context_window': 8192,
        'cache_bit': 16,
        'num_attention_heads': 128,
        'num_key_value_heads': 128,
        'hidden_size': 7168,
        'num_hidden_layers': 61,
        'selected_llm': 'DeepSeek-R1',
    }

    request.session["model_config_initial"] = model_config_initial
    model_form = ModelConfigurationForm(request.POST or None, initial=model_config_initial)

    # Instantiate both OS-specific forms with initial values.
    mac_form = MacOsSystemInformationForm(
        request.POST or None,
        initial={
            "macos_unified_ram": 24,  
            "m_series_processor": "M1"  
        }
    )

    win_form = WindowsSystemInformationForm(
        request.POST or None,
        initial={
            "windows_gpu_vram": 16,
            "windows_ram": 8
        }
    )
    
    # Default to Windows if nothing is posted
    os_choice = "win"
    
    # Prepare the base context.
    context = {
        'model_form': model_form,
        'mac_form': mac_form,
        'win_form': win_form,
        'llm_choices_json': json.dumps(get_llm_choices_mapping()),
        'os_choice': os_choice
    }
    
    tks = None
    offload_ratio = None

    mac_processors = AppleMSeriesProcessor.objects.all()
    processor_mapping = {proc.name: proc.bandwidth for proc in mac_processors}

    # Process the POST request without using cleaned_data.
    if request.method == 'POST':
        # Determine which OS button was clicked.
        if "mac" in request.POST:
            print("Mac")
            os_choice = "mac"
            win_form = WindowsSystemInformationForm(request.POST)
            mac_form = MacOsSystemInformationForm(initial={
                "macos_unified_ram": 24,
                "m_series_processor": "M1"
            })

        elif "windows" in request.POST:
            os_choice = "win"
            mac_form = MacOsSystemInformationForm(request.POST)
            win_form = WindowsSystemInformationForm(initial={
                "windows_gpu_vram": 16,
                "windows_ram": 8
            })

        else:
            # Fallback: use session or default
            os_choice = request.session.get("os_choice", "win")

        # Use the raw form data via form.data instead of cleaned_data.
        if os_choice == 'mac':
            # Get Mac values from the bound mac_form
            unified_ram = mac_form.data.get('macos_unified_ram', 24)
            m_processor = mac_form.data.get('m_series_processor', "M1")

            try:
                unified_ram = float(unified_ram)
            except (ValueError, TypeError):
                unified_ram = 24

            apple_processor = AppleMSeriesProcessor.objects.get(name=m_processor)
            mac_bandwidth = apple_processor.bandwidth
            system_vram = unified_ram   # For mac, unified memory is used as VRAM.
            system_ram = 0              # No separate system RAM.
            gpu_bandwidth = mac_bandwidth
            ram_bandwidth = mac_bandwidth

            print("\n----------------------------")
            print("Os: MACOS")
            print(f"Processor: {m_processor}")
            print(f"Bandwidth: {mac_bandwidth}")
            print(f"Unified RAM: {system_vram}")
            print("----------------------------\n")

        elif os_choice == 'win':
            # Get Windows values from the bound win_form.
            system_ram = win_form.data.get('windows_ram', 8)
            system_vram = win_form.data.get('windows_gpu_vram', 16)

            # Convert to numeric types:
            try:
                system_ram = float(system_ram)
            except (ValueError, TypeError):
                system_ram = 8

            try:
                system_vram = float(system_vram)
            except (ValueError, TypeError):
                system_vram = 16

            try:
                gpu_bandwidth = float(win_form.data.get('windows_gpu_bandwidth', 0))
                ram_bandwidth = float(win_form.data.get('windows_ram_bandwidth', 0))
            except:
                gpu_bandwidth = None
                ram_bandwidth = None

            print("\n----------------------------")
            print("Os: Windows")
            print(f"RAM: {system_ram}")
            print(f"VRAM: {system_vram}")
            print(f"GPU Bandwidth: {gpu_bandwidth}")
            print(f"RAM Bandwidth: {ram_bandwidth}")
            print("----------------------------\n")

        if model_form.is_valid():        
            parameters_model    = model_form.cleaned_data.get('parameters_model', 685)
            quantization_level  = model_form.cleaned_data.get('quantization_level', 'fp16')
            context_window      = model_form.cleaned_data.get('context_window', 8192)
            cache_bit           = model_form.cleaned_data.get('cache_bit', 16)
            num_attention_heads = model_form.cleaned_data.get('num_attention_heads', 128)
            num_key_value_heads = model_form.cleaned_data.get('num_key_value_heads', 128)
            hidden_size         = model_form.cleaned_data.get('hidden_size', 7168)
            num_hidden_layers   = model_form.cleaned_data.get('num_hidden_layers', 61)
            configuration_mode  = model_form.cleaned_data.get('configuration_mode', "simple")

        
        # Save values to session.
        session_data = {
            "ram": system_ram if os_choice == 'win' else None,
            "vram": system_vram if os_choice == 'win' else None,
            "gpu_bandwidth": gpu_bandwidth if os_choice == 'win' else None,
            "ram_bandwidth": ram_bandwidth if os_choice == 'win' else None,
            "mac_bandwidth": mac_bandwidth if os_choice == 'mac' else None,
            "unified_ram": unified_ram if os_choice == 'mac' else None,
            "m_processor": m_processor if os_choice == 'mac' else None,
            "configuration_mode": configuration_mode,
            "os_choice": os_choice,
        }

        print("\n----------------------------")
        print(f"Session data:")
        for key, value in session_data.items():
            print(f"{key}: {value}")
        print("----------------------------\n")
     

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

        if configuration_mode == "simple":
            total_vram = vram_calculator.compute_vram_simple()

        elif configuration_mode == "advanced":
            total_vram = vram_calculator.compute_vram_advanced()

        else:
            total_vram = vram_calculator.compute_vram_intermediate()

        model_weight_vram = vram_calculator.model_weights()
        kv_cache_vram = vram_calculator.kv_cache()
        cuda_buffer_vram = vram_calculator.cuda_buffer()

        can_run_result = None

        if "run_check" in request.POST:
            if gpu_bandwidth and ram_bandwidth:
                estimator = EstimateTokenPerSecond(
                    gpu_bandwidth,
                    ram_bandwidth,
                    system_vram,
                    system_ram,
                    quantization_level,
                    total_vram
                )
                tks, _ = estimator.calculate_token_per_second()

            offload_ratio = round(((total_vram - system_vram) / total_vram * 100), 2)

            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram if system_vram is not None else 0,
                system_ram=system_ram if system_ram is not None else 0
            )

            can_run_result = checker.decide()

        elif "stop_light_chart" in request.POST:
            return HttpResponseRedirect(reverse("stop_chart"))

        # Update the context with calculated results.
        context["model_form"] = ModelConfigurationForm(initial=model_form.data)
        context.update({
            'vram': total_vram,
            'mode_weight': model_weight_vram,
            'kv_cache': kv_cache_vram,
            'cuda_buffer': cuda_buffer_vram,
            'can_run_result': can_run_result,
            'tks': tks,
            'offload_ratio': offload_ratio,
            'os_choice': os_choice,
            'mac_form': mac_form if mac_form is not None else mac_form,
            'win_form': win_form if win_form is not None else win_form,
            'processor_mapping': json.dumps(processor_mapping) 

        })

        return render(request, 'System/home.html', context)

    return render(request, 'System/home.html', context)