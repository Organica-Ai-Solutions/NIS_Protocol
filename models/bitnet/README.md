# BitNet Models

This directory contains BitNet model configurations and scripts, but not the actual model weights due to their large size (4.4GB+).

## Getting the Model Files

To use the BitNet models, you have several options:

### Option 1: Use the Download Script (Recommended)

We provide a convenient script to download the BitNet model:

```bash
# From the project root directory
python scripts/download_bitnet_models.py

# To specify a different model or output directory
python scripts/download_bitnet_models.py --model microsoft/BitNet --output custom/path/to/models
```

### Option 2: Download from Hugging Face Hub in Your Code

Install the transformers library and download the model using the Hugging Face Hub:

```python
from transformers import AutoModel, AutoTokenizer

# This will automatically download and cache the model
model_name = "microsoft/BitNet"  # Replace with the specific model you need
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### Option 3: Manual Download

1. Download the model files from [Hugging Face Microsoft/BitNet](https://huggingface.co/microsoft/BitNet)
2. Place the downloaded files in the following directory structure:
   ```
   models/bitnet/models/bitnet/
   ```

## Model Files Structure

After downloading, your directory structure should look like:

```
models/bitnet/
├── models/
│   ├── bitnet/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── model.safetensors  # Large file, not in git repository
│   │   └── ...
```

## Note on Git Repository

The actual model files (*.safetensors, *.bin, etc.) are excluded from the Git repository due to their large size. They should be downloaded separately as described above.

## Usage in the NIS Protocol

```python
from src.llm.providers.bitnet_provider import BitNetProvider

bitnet = BitNetProvider()
response = bitnet.generate("Your prompt here")
``` 