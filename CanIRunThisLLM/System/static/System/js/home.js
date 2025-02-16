document.addEventListener("DOMContentLoaded", function () {
    const modeField = document.getElementById("id_configuration_mode");
    const advancedContainer = document.getElementById("advanced-fields");

    function toggleAdvancedFields() {
        if (modeField.value === "advanced") {
            advancedContainer.style.display = "block";
        } else {
            advancedContainer.style.display = "none";
        }
    }

    toggleAdvancedFields();
    modeField.addEventListener("change", toggleAdvancedFields);
});

document.addEventListener("DOMContentLoaded", function () {
    const modelSelector = document.getElementById("id_selected_llm");
    // Instead of trying to parse a template variable,
    // use the global variable defined in your HTML:
    const llmMapping = window.llmMapping;
    if (modelSelector) {
        modelSelector.addEventListener("change", function () {
            const selectedModel = modelSelector.value;
            if (selectedModel in llmMapping) {
                const config = llmMapping[selectedModel];
                document.getElementById("id_parameters_model").value = config.parameters / 1000000000;
                document.getElementById("id_quantization_level").value = config.quant_level;
                document.getElementById("id_context_window").value = config.context_window;
                document.getElementById("id_cache_bit").value = config.cache_bit;
                document.getElementById("id_num_attention_heads").value = config.model_config.num_attention_heads;
                document.getElementById("id_num_key_value_heads").value = config.model_config.num_key_value_heads;
                document.getElementById("id_hidden_size").value = config.model_config.hidden_size;
                document.getElementById("id_num_hidden_layers").value = config.model_config.num_hidden_layers;
            }
        });
    }
});


