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


  // Suppose you have a way to get system info on the client
  // For instance, using limited data from the browser:
  const systemInfo = {
    userAgent: navigator.userAgent,
    screenWidth: window.screen.width,
    screenHeight: window.screen.height,
    cpuCores: navigator.hardwareConcurrency || 'N/A'
  };

  // Send data via fetch so it's attached to the browser's session
  fetch('/upload/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': '{{ csrf_token }}'
    },
    body: JSON.stringify(systemInfo)
  })
  .then(response => {
    if (response.redirected) {
      window.location.href = response.url;
    }
  })
  .catch(error => console.error('Error sending system info:', error));

  document.addEventListener('DOMContentLoaded', function () {
    const macBtn = document.getElementById('mac-btn');
    const windowsBtn = document.getElementById('windows-btn');
    const osChoice = document.getElementById('os-choice');
  
    macBtn.addEventListener('click', function() {
      windowsBtn.disabled = true;
      macBtn.classList.add('selected');
      osChoice.value = 'mac';
      macBtn.focus();  // Give focus to the clicked button
    });
  
    windowsBtn.addEventListener('click', function() {
      macBtn.disabled = true;
      windowsBtn.classList.add('selected');
      osChoice.value = 'windows';
      windowsBtn.focus();  // Give focus to the clicked button
    });
  });
  
document.addEventListener('DOMContentLoaded', function () {
  const macBtn = document.getElementById('mac-btn');
  const windowsBtn = document.getElementById('windows-btn');
  const osChoice = document.getElementById('os-choice');
  const appleForm = document.getElementById('apple-form');
  const defaultForm = document.getElementById('default-form');

  macBtn.addEventListener('click', function() {
  // Update the hidden input
  osChoice.value = 'mac';
  // Show the Apple-specific form and hide the default (Windows) form
  appleForm.style.display = 'block';
  defaultForm.style.display = 'none';
  // Add selected class to Mac and remove it from Windows
  macBtn.classList.add('selected');
  windowsBtn.classList.remove('selected');
  macBtn.focus();
  });

  windowsBtn.addEventListener('click', function() {
  // Update the hidden input
  osChoice.value = 'windows';
  // Show the default (Windows) form and hide the Apple-specific form
  defaultForm.style.display = 'block';
  appleForm.style.display = 'none';
  // Add selected class to Windows and remove it from Mac
  windowsBtn.classList.add('selected');
  macBtn.classList.remove('selected');
  windowsBtn.focus();
  });
});