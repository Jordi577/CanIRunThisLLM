# CanYourPcRunThisLLM

## Introduction
This web project allows you to estimate the VRAM requirements of local models. Enter your system's VRAM and RAM specifications to determine if it can run the chosen local model. Additionally, provide your GPU and RAM bandwidth information to estimate the tokens processed per second.

Visit the website: [http://www.canirunthisllm.net/](http://www.canirunthisllm.net/)

![grafik](https://github.com/user-attachments/assets/a6960415-4d30-4187-a558-5725b211221f)

If you'd like to see a more detailed chart, just press the icon in the System Information card.
![grafik](https://github.com/user-attachments/assets/be4c64bd-5a3c-41c6-96e6-bdec20f70e11)

## Setup Instructions

To run this project locally, follow these steps:

### 1. Install Poetry (if not already installed)
Poetry is a dependency management tool for Python. Install it using:

```bash
pip install poetry
```

### 2. Configure a Virtual Environment
Ensure that Poetry creates a virtual environment within the project directory:

```bash
poetry config virtualenvs.in-project true
```

This sets up a `.venv` directory inside the repository.

### 3. Install Dependencies
Run the following command to install all required dependencies:

```bash
poetry install --no-root
```

Once installed, you’re ready to start working with the project!

## Running the Server

To start the server, navigate to the project directory and run:

```bash
cd CanMyPcRunIt
python manage.py runserver
```

This will launch the development server, allowing you to interact with the application locally.

## How does the VRAM get calculated?
The VRAM estimation is computed using multiple components related to the model’s architecture and configuration. Below are the primary factors considered in the calculation:

1. **Model Weights VRAM**
   - The model parameters are multiplied by a quantization factor that depends on the precision level (e.g., fp32, fp16, q8, q4, etc.).
   - Lower precision levels reduce memory usage by storing fewer bits per parameter.
   - The formula used: 
     ```
     VRAM = Parameters * Bytes_Per_Weight[quant_level]
     ```

2. **KV Cache VRAM**
   - The KV (Key-Value) cache is calculated based on the number of attention heads, key-value heads, hidden size, and context window.
   - The formula considers grouped query attention and determines the cache size in GB.
   - The formula used:
     ```
     KV_Cache = 2 * n_elements * (cache_bit / 8) / 1e9
     ```

3. **CUDA Buffer**
   - An overhead buffer is added to account for CUDA memory management, which is set by default to 0.5GB.

4. **Total VRAM Calculation**
   - The total VRAM requirement is the sum of the model weights, KV cache, and CUDA overhead.
   - The formula used:
     ```
     Total_VRAM = Model_Weights + KV_Cache + CUDA_Overhead
     ```
