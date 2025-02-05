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

    print("System information stored in session:",
          request.session.get("system_information"))
    return HttpResponseRedirect(reverse('home'))


def update_table_view(request):
    """
    This view receives GET parameters 'system_ram' and 'system_vram',
    calculates the table data, and returns the updated table HTML.
    """
    try:
        system_ram = float(request.GET.get('system_ram', 0))
        system_vram = float(request.GET.get('system_vram', 0))
    except ValueError:
        system_ram = 0
        system_vram = 0

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
                context_window=value["context_window"],
                cache_bit=value["cache_bit"]
            )

            if configuration_mode == "simple":
                total_vram = vram_calculator.compute_vram_simple()
            elif configuration_mode == "advanced":
                total_vram = vram_calculator.compute_vram_advanced()
            else:
                total_vram = vram_calculator.compute_vram_intermediate()

            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            )
            can_run_result = checker.decide()
            row_result["values"].append(
                (colors_map[can_run_result], round(total_vram, 1)))
        result.append(row_result)

    context = {
        "columns": columns,
        "chart_data": result,
    }
    html = render_to_string("System/partials/table.html", context)
    return HttpResponse(html)


def stop_chart_view(request):
    configuration_mode = request.session.get("configuration_mode", "simple")
    system_vram = request.session.get("vram", 0)
    system_ram = request.session.get("ram", 0)
    initial = request.session.get("initial")

    if "return_home" in request.POST:
        initial = request.session.get("initial")
        return HttpResponseRedirect(reverse("home"))

    print(initial)
    initial = {"system_ram": system_ram, "system_vram": system_vram}
    system_spec = SystemInformation(request.POST or None, initial=initial)

    if "calculate" in request.POST:
        print("hh")
        print(system_spec.errors)
        if system_spec.is_valid():
            print("hhww")
            system_ram = system_spec.cleaned_data['system_ram']
            system_vram = system_spec.cleaned_data['system_vram']
            print(system_vram)

    rows = []
    columns = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "fp16", "fp32"]
    colors_map = {1: "green", 2: "yellow", 3: "red"}

    result = []

    for model, value in LLM_CHOICES_MAPPING.items():
        row_result = {"row": model, "values": []}
        rows.append(model)

        for quant in columns:
            vram_calculator = ModelVRAMCalculator(
                model_config={
                    "num_attention_heads": value["model_config"]["num_attention_heads"],
                    "num_key_value_heads": value["model_config"]["num_key_value_heads"],
                    "hidden_size": value["model_config"]["hidden_size"],
                    "num_hidden_layers": value["model_config"]["num_hidden_layers"],
                },
                parameters=value["parameters"]/1e9,
                quant_level=quant,
                context_window=value["context_window"],
                cache_bit=value["cache_bit"]
            )

            if configuration_mode == "simple":
                total_vram = vram_calculator.compute_vram_simple()

            elif configuration_mode == "advanced":
                total_vram = vram_calculator.compute_vram_advanced()

            else:
                total_vram = vram_calculator.compute_vram_intermediate()

            checker = CanIRunIt(
                required_vram=total_vram,
                system_vram=system_vram,
                system_ram=system_ram
            )
            can_run_result = checker.decide()
            row_result["values"].append(
                (colors_map[can_run_result], round(total_vram, 1)))

        result.append(row_result)

    stop_light_chart = result

    context = {
        "columns": columns,
        "chart_data": stop_light_chart,
        "ram": system_ram,
        "vram": system_vram,
        "system_form": system_spec
    }

    return render(request, "System/stop_chart.html", context)


def home(request):
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
    main_form = VRAMCalculationForm(request.POST or None)
    hf_form = HuggingfaceModelForm(request.POST or None)

    if "return" in request.POST:
        main_form = VRAMCalculationForm(initial=initial)
        return render(request, 'System/home.html', {
            'form': main_form,
            'hf_form': hf_form,
            'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
        })

    if request.method == 'POST':
        if "hf_model" in request.POST:
            if hf_form.is_valid():
                model_path = hf_form.cleaned_data.get("huggingface_model_path")
                predefined_custom = hf_form.cleaned_data.get(
                    "predefined_model_custom")
                # Decide which value to use (you might want to prefer one over the other).
                chosen_model = model_path or predefined_custom
                print("Huggingface Model Path:", chosen_model)

                ########## Retreiving model card from hugingface ##########
                try:
                    website_url = "https://huggingface.co/" + model_path
                    model_name = model_path.split('/')[-1]
                    extractor = ModelExtractor(url=website_url)
                    extractor.fetch_data_from_website()
                    model_size_number = extractor.extract_model_size_number()
                    config_file_path = extractor.download_model_config()
                    final_config = extractor.build_final_config(
                        config_file_path, model_name, model_size_number)
                    print(final_config)
                    print("---")
                    print(initial)

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
                    main_form = VRAMCalculationForm(initial=initial)
                    return render(request, 'System/home.html', {
                        'form': main_form,
                        'hf_form': hf_form,
                        'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
                    })

                except:
                    main_form = VRAMCalculationForm(initial=initial)
                    print("hi")
                    return render(request, 'System/home.html', {
                        'form': main_form,
                        'hf_form': hf_form,
                        'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
                        'error': True
                    })

        if main_form.is_valid():
            parameters_model = main_form.cleaned_data['parameters_model']
            quantization_level = main_form.cleaned_data['quantization_level']
            context_window = main_form.cleaned_data['context_window']
            cache_bit = main_form.cleaned_data['cache_bit']
            num_attention_heads = main_form.cleaned_data['num_attention_heads']
            num_key_value_heads = main_form.cleaned_data['num_key_value_heads']
            hidden_size = main_form.cleaned_data['hidden_size']
            num_hidden_layers = main_form.cleaned_data['num_hidden_layers']
            configuration_mode = main_form.cleaned_data.get(
                "configuration_mode", "simple")

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

            manual_ram = main_form.cleaned_data.get('ram')
            manual_vram = main_form.cleaned_data.get('gpu_vram')

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
                checker = CanIRunIt(
                    required_vram=total_vram,
                    system_vram=manual_vram if manual_vram is not None else 0,
                    system_ram=manual_ram if manual_ram is not None else 0
                )
                can_run_result = checker.decide()

            elif "stop_light_chart" in request.POST:
                request.session["configuration_mode"] = main_form.cleaned_data.get(
                    "configuration_mode", "simple")
                request.session["vram"] = manual_vram if manual_vram is not None else 0
                request.session["ram"] = manual_ram if manual_ram is not None else 0

                return HttpResponseRedirect(reverse("stop_chart"))

            main_form = VRAMCalculationForm(initial=main_form.cleaned_data)

            return render(request, 'System/home.html', {
                'form': main_form,
                'hf_form': hf_form,
                'vram': total_vram,
                'mode_weight': model_weight_vram,
                'kv_cache': kv_cache_vram,
                'cuda_buffer': cuda_buffer_vram,
                'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
                'can_run_result': can_run_result,
                'ram': manual_ram,
                'gpu_vram': manual_vram,
            })

    else:
        main_form = VRAMCalculationForm(initial=initial)
        hf_form = HuggingfaceModelForm()

    return render(request, 'System/home.html', {
        'form': main_form,
        'hf_form': hf_form,
        'llm_choices_json': json.dumps(LLM_CHOICES_MAPPING),
    })