from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from .static.vram_calc import ModelVRAMCalculator
from .static.canirunit import CanIRunIt
from django.urls import reverse
from django.http import HttpResponseRedirect
from ModelExtractor.extractor import ModelExtractor
from django.template.loader import render_to_string
from .forms import VRAMCalculationForm, HuggingfaceModelForm, LLM_CHOICES_MAPPING, SystemInformation
import json

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
        system_ram = 0
        system_vram = 0
        system_context_window = 0

    configuration_mode = request.session.get("configuration_mode", "simple")
    columns = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}

    result = []
    for model, value in LLM_CHOICES_MAPPING.items():
        row_result = {"row": model, "values": []}

        for quant in columns:
            vram_calculator = ModelVRAMCalculator(
                model_config={
                    "num_attention_heads": value["model_config"]["num_attention_heads"],
                    "num_key_value_heads": value["model_config"]["num_key_value_heads"],
                    "hidden_size": value["model_config"]["hidden_size"],
                    "num_hidden_layers": value["model_config"]["num_hidden_layers"],
                },
                parameters=value["parameters"] / 1e9,
                quant_level=quant,
                context_window=system_context_window,
                cache_bit=value["cache_bit"]
            )

            # if configuration_mode == "simple":
            #     total_vram = vram_calculator.compute_vram_simple()

            # elif configuration_mode == "advanced":
            #     total_vram = vram_calculator.compute_vram_advanced()

            # else:
            #     total_vram = vram_calculator.compute_vram_intermediate()

            total_vram = vram_calculator.compute_vram_advanced()
            checker = CanIRunIt(required_vram=total_vram, system_vram=system_vram, system_ram=system_ram)
            can_run_result = checker.decide()
            row_result["values"].append((colors_map[can_run_result], round(total_vram, 1)))

        result.append(row_result)

    context = {"columns": columns, "chart_data": result}
    html = render_to_string("System/partials/table.html", context)

    return HttpResponse(html)


def stop_chart_view(request):
    # Retrieve system configuration from the session
    config_mode = request.session.get("configuration_mode", "simple")
    system_vram = request.session.get("vram", 0)
    system_ram = request.session.get("ram", 0)
    context_window = 8192

    # Initialize the system form with the current system RAM and VRAM values
    initial_data = {
        "system_ram": system_ram,
        "system_vram": system_vram,
        "context_window": context_window
        }

    system_form = SystemInformation(request.POST or None, initial=initial_data)

    # If the user requested to return home, redirect immediately
    if "return_home" in request.POST:
        return HttpResponseRedirect(reverse("home"))

    # Define quantization levels and color mappings for the chart
    quant_levels = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}

    chart_data = []

    # Process each model in the LLM_CHOICES_MAPPING
    for model_name, model_info in LLM_CHOICES_MAPPING.items():
        row_data = {"row": model_name, "values": []}

        for quant_level in quant_levels:
            model_config = model_info["model_config"]

            # Create a VRAM calculator for the model with the given quantization level
            vram_calculator = ModelVRAMCalculator(
                model_config={
                    "num_attention_heads": model_config["num_attention_heads"],
                    "num_key_value_heads": model_config["num_key_value_heads"],
                    "hidden_size": model_config["hidden_size"],
                    "num_hidden_layers": model_config["num_hidden_layers"],
                },
                parameters=model_info["parameters"] / 1e9,
                quant_level=quant_level,
                context_window=model_info["context_window"],
                cache_bit=model_info["cache_bit"]
            )

            # Select the appropriate VRAM computation method based on the configuration mode
            if config_mode == "simple":
                total_vram = vram_calculator.compute_vram_simple()
            elif config_mode == "advanced":
                total_vram = vram_calculator.compute_vram_advanced()
            else:
                total_vram = vram_calculator.compute_vram_intermediate()

            # Determine if the system meets the VRAM requirements
            can_run = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            ).decide()

            # Append the result with the corresponding color and rounded VRAM value
            row_data["values"].append((colors_map[can_run], round(total_vram, 1)))

        chart_data.append(row_data)

    # Build the context for rendering the chart template
    context = {
        "columns": quant_levels,
        "chart_data": chart_data,
        "ram": system_ram,
        "vram": system_vram,
        "system_form": system_form
    }

    return render(request, "System/stop_chart.html", context)


def home(request):
    # Default initial parameters.
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
    request.session["initial"] = initial

    # Instantiate the forms.
    main_form = VRAMCalculationForm(request.POST or None, initial=initial)
    hf_form = HuggingfaceModelForm(request.POST or None)

    # Base context to pass to the template.
    context = {
        'form': main_form,
        'hf_form': hf_form,
        'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
    }

    # Handle a "return" request (resetting the main form to initial values).
    if "return" in request.POST:
        context["form"] = VRAMCalculationForm(initial=initial)
        return render(request, 'System/home.html', context)

    if request.method == 'POST':
        # Handle the Huggingface Model form.
        if "hf_model" in request.POST:
            if hf_form.is_valid():
                model_path = hf_form.cleaned_data.get("huggingface_model_path")
                predefined_custom = hf_form.cleaned_data.get("predefined_model_custom")
                chosen_model = model_path or predefined_custom
                print("Huggingface Model Path:", chosen_model)

                try:
                    website_url = "https://huggingface.co/" + model_path
                    model_name = model_path.split('/')[-1]
                    extractor = ModelExtractor(url=website_url)
                    extractor.fetch_data_from_website()
                    model_size_number = extractor.extract_model_size_number()
                    config_file_path = extractor.download_model_config()
                    final_config = extractor.build_final_config(
                        config_file_path, model_name, model_size_number
                    )
                    print(final_config)
                    print("---")
                    print(initial)

                    # Update initial parameters based on the fetched configuration.
                    initial = {
                        'parameters_model': final_config["parameters"],
                        'quantization_level': 'fp16',
                        'context_window': 8192,
                        'cache_bit': 16,
                        'num_attention_heads': final_config["model_config"]["num_attention_heads"],
                        'num_key_value_heads': final_config["model_config"]["num_key_value_heads"],
                        'hidden_size': final_config["model_config"]["hidden_size"],
                        'num_hidden_layers': final_config["model_config"]["num_hidden_layers"],
                        'selected_llm': final_config["name"],
                        'ram': 16,
                        'gpu_vram': 8
                    }
                    request.session["initial"] = initial
                    context["form"] = VRAMCalculationForm(initial=initial)
                    return render(request, 'System/home.html', context)

                except Exception as e:
                    print("Error processing Huggingface model:", e)
                    context["form"] = VRAMCalculationForm(initial=initial)
                    context["error"] = True
                    return render(request, 'System/home.html', context)

        # Handle the main VRAM calculation form.
        if main_form.is_valid():
            cd = main_form.cleaned_data
            # Extract values from the cleaned data.
            parameters_model = cd['parameters_model']
            quantization_level = cd['quantization_level']
            context_window = cd['context_window']
            cache_bit = cd['cache_bit']
            num_attention_heads = cd['num_attention_heads']
            num_key_value_heads = cd['num_key_value_heads']
            hidden_size = cd['hidden_size']
            num_hidden_layers = cd['num_hidden_layers']
            configuration_mode = cd.get("configuration_mode", "simple")

            # Create the VRAM calculator.
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

            # Compute other VRAM metrics.
            model_weight_vram = vram_calculator.model_weights()
            kv_cache_vram = vram_calculator.kv_cache()
            cuda_buffer_vram = vram_calculator.cuda_buffer()

            manual_ram = cd.get('ram')
            manual_vram = cd.get('gpu_vram')

            # Check if the system can run the model.
            can_run_result = None
            if "run_check" in request.POST:
                checker = CanIRunIt(
                    required_vram=total_vram,
                    system_vram=manual_vram if manual_vram is not None else 0,
                    system_ram=manual_ram if manual_ram is not None else 0
                )
                can_run_result = checker.decide()

            # Redirect to the stop light chart if requested.
            elif "stop_light_chart" in request.POST:
                request.session["configuration_mode"] = configuration_mode
                request.session["vram"] = manual_vram if manual_vram is not None else 0
                request.session["ram"] = manual_ram if manual_ram is not None else 0
                return HttpResponseRedirect(reverse("stop_chart"))

            # Reinitialize the form with the cleaned data.
            context["form"] = VRAMCalculationForm(initial=cd)

            # Update the context with the computed VRAM values and check result.
            context.update({
                'vram': total_vram,
                'mode_weight': model_weight_vram,
                'kv_cache': kv_cache_vram,
                'cuda_buffer': cuda_buffer_vram,
                'can_run_result': can_run_result,
                'ram': manual_ram,
                'gpu_vram': manual_vram,
            })

            return render(request, 'System/home.html', context)

    # For non-POST requests, initialize forms with
    context["form"] = VRAMCalculationForm(initial=initial)
    context["hf_form"] = HuggingfaceModelForm()

    return render(request, 'System/home.html', context)