# Self-Hosted OpenAI-Compatible Gemma 3 Chat Completion Endpoint

This project enables you to self-host an OpenAI-compatible chat completion API endpoint powered by one of the google gemma3 models - defaults to the [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) model. Deployed via a FastAPI server within a Docker container, this solution supports multimodal inputs with a focus on text and images, leveraging NVIDIA GPU acceleration for efficient inference.

Tested on AWS on G6 instance

## Features

- **Multimodal Capabilities**: Process text and image inputs seamlessly
- **OpenAI Compatibility**: Integrate easily with the OpenAI Python client or any HTTP client
- **GPU Support**: Utilizes NVIDIA GPUs with automatic model distribution across available devices
- **Persistent Caching**: Stores model weights in a Docker volume to avoid repeated downloads
- **Advanced Generation Parameters**: Supports fine-tuned control over generation with parameters like temperature, top_p, top_k, min_p, and repetition penalty
- **Health Monitoring**: Built-in health check endpoint

## Prerequisites

Before you begin, ensure the following are installed on your system:

- **[Docker](https://docs.docker.com/get-docker/)**: For containerization
- **[Docker Compose](https://docs.docker.com/compose/install/)**: For managing multi-container setups
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**: For GPU support within Docker
- **Compatible NVIDIA GPU(s)**: With at least 24GB VRAM per GPU for optimal performance
- **[Hugging Face Account](https://huggingface.co/join)**: To access the model weights (free account required)

## Setup Instructions

Follow these steps to build and run the self-hosted endpoint.

### 1. Clone the Repository

```bash
git clone https://github.com/anastasiosyal/google-gemma3-inference-server.git
cd google-gemma3-inference-server
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required: Your Hugging Face token for downloading model weights
HF_TOKEN=your_hugging_face_token_here

# Optional: Change the model (defaults to google/gemma-3-12b-it)
MODEL_ID=google/gemma-3-12b-it
```

You can get your Hugging Face token from your [account settings](https://huggingface.co/settings/tokens).

### 3. Start the Service

Simply run:

```bash
docker compose up -d
```

This command will:
- Build the container if needed
- Mount necessary volumes for model caching
- Configure GPU access
- Set up environment variables
- Start the service in detached mode

The API will be available at `http://localhost:8100`.

To view the logs:

```bash
docker compose logs -f
```

To stop the service:

```bash
docker compose down
```

### 4. Test the Model Directly

You can test the model directly using the official Hugging Face example script included in the container:

```bash
# Connect to the running container
docker compose exec -it gemma3-server bash

# Navigate to the app directory (where test_model.py is located)
cd /app

# Run the test script
python3 test_model.py
```

This will run the example script from the Gemma 3 official Hugging Face repository, which demonstrates the model's basic capabilities without going through the API endpoint. It's a good way to verify that:
- The model loads correctly
- Your GPU setup is working
- Token authentication is successful
- Basic inference is working as expected

## Usage

Interact with the endpoint using the OpenAI Python client or any HTTP client. Below are examples demonstrating different types of requests.

### Example: Text Request

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="test"  # API key not required in current implementation
)

response = client.chat.completions.create(
    model="google/gemma-3-8b-it",
    messages=[{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Explain quantum computing in simple terms."
        }]
    }],
    max_completion_tokens=1000,
    temperature=0.7,
    top_p=0.95,
    top_k=64,
    min_p=0.01,
    repetition_penalty=1.0
)

print(response.choices[0].message.content)
```

### Example: Text and Image Request

```python
import openai
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "image.jpg"
image_base64 = encode_image(image_path)

client = openai.OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="test"
)

response = client.chat.completions.create(
    model="google/gemma-3-8b-it",
    messages=[{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Describe this image in detail."
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }]
    }],
    max_completion_tokens=1000
)

print(response.choices[0].message.content)
```

### API Parameters

The endpoint supports the following parameters for fine-tuned control:

- `max_completion_tokens` (default: 1000): Maximum number of tokens to generate
- `temperature` (default: 1.0): Controls randomness in generation
- `top_p` (default: 0.95): Nucleus sampling parameter
- `top_k` (default: 64): Top-k sampling parameter
- `min_p` (default: 0.01): Minimum probability for token selection
- `repetition_penalty` (default: 1.0): Penalty for repeating tokens

### Health Check

Monitor the service status using the health endpoint:

```bash
curl http://localhost:8100/health
```

## Under the Hood

The system architecture includes several key components:

### 1. **Model Configuration**

- Uses the Gemma 3 8B instruction-tuned model
- Automatically distributes model across available GPUs using HuggingFace's `device_map="auto"`
- Utilizes bfloat16 precision for optimal performance
- Configures 24GB memory limit per GPU

### 2. **FastAPI Server**

- Request logging and error handling
- Support for system prompts and chat history

### 3. **Performance Optimizations**

- Flash Attention 2.7.4 for efficient attention computation
- Automatic model sharding across multiple GPUs
- Persistent caching of model weights

## Troubleshooting

- **GPU Memory Issues**: Ensure you have sufficient VRAM (24GB per GPU recommended)
- **Model Loading Errors**: 
  - Verify your Hugging Face token is correct in the `.env` file
  - Check the logs for detailed error messages using `docker compose logs -f`
  - Verify HuggingFace cache permissions
- **Generation Issues**: Verify input format and adjust generation parameters as needed
- **Performance Problems**: Monitor GPU utilization and adjust worker count if necessary

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is released under the [MIT License](LICENSE).
