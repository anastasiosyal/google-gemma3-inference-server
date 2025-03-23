from fastapi import FastAPI, Request
import requests
from PIL import Image
import base64
from io import BytesIO
import tempfile
import soundfile as sf
import logging
import sys
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
import json
from fastapi.responses import JSONResponse  # This was missing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add middleware to log request details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request details and catch errors."""
    request_id = str(id(request))
    client_host = request.client.host if request.client else "unknown"
    
    # Log request details
    logger.info(f"Request {request_id} started: {request.method} {request.url.path} from {client_host}")
    
    # Log request headers
    headers_log = dict(request.headers)
    # Redact sensitive headers
    if "authorization" in headers_log:
        headers_log["authorization"] = "***REDACTED***"
    logger.info(f"Request {request_id} headers: {headers_log}")
    
    # Try to log body for debugging, but don't fail if we can't
    try:
        # Save the request body
        body_bytes = await request.body()
        if body_bytes:
            # Try to parse as JSON for better logging
            try:
                body_json = json.loads(body_bytes)
                # Truncate potentially large content fields
                if "messages" in body_json:
                    for msg in body_json["messages"]:
                        if "content" in msg and isinstance(msg["content"], list):
                            for item in msg["content"]:
                                if item.get("type") == "text" and len(item.get("text", "")) > 100:
                                    item["text"] = item["text"][:100] + "... [truncated]"
                                if item.get("type") == "image_url" and "url" in item.get("image_url", {}):
                                    item["image_url"]["url"] = "[IMAGE URL]"
                logger.info(f"Request {request_id} body: {json.dumps(body_json)}")
            except json.JSONDecodeError:
                # Not JSON, log first 200 chars
                body_str = body_bytes.decode('utf-8', errors='replace')
                logger.info(f"Request {request_id} body (non-JSON): {body_str[:200]}")
        
        # Forward the request and get response
        try:
            response = await call_next(request)
            logger.info(f"Request {request_id} completed with status {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Request {request_id} failed with error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500, 
                content={"error": str(e), "detail": traceback.format_exc()}
            )
    except Exception as middleware_error:
        # If our logging fails, don't prevent the request from being processed
        logger.error(f"Error in request logging middleware: {str(middleware_error)}")
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Request failed with error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})


# Define model path from environment variable with fallback
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-8b-it")  # Use 8B as default, configurable via env

# Load model and processor
logger.info(f"Loading model {MODEL_ID}...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Configure device map to utilize all 4 GPUs
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",  # Will distribute model across available GPUs
        torch_dtype=torch.bfloat16,
        max_memory={i: "24GiB" for i in range(torch.cuda.device_count())},  # Adjust memory per GPU if needed
    ).eval()
    logger.info(f"Model loaded successfully: {MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    processor = None
    model = None

def process_message_content(content_list):
    messages = []
    for item in content_list:
        if item["type"] == "text":
            messages.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            image_url = item["image_url"]["url"]
            if image_url.startswith("data:image/"):
                # It's base64-encoded
                parts = image_url.split(",")
                if len(parts) == 2:
                    image_data = base64.b64decode(parts[1])
                    image = Image.open(BytesIO(image_data))
                else:
                    raise ValueError("Invalid base64 image URL")
            else:
                # It's a regular URL
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            
            image = image.convert('RGB')  # Ensure RGB mode
            logger.info(f"Loaded image, size: {image.size}, mode: {image.mode}")
            messages.append({"type": "image", "image": image})
            
        elif item["type"] == "input_audio":
            data = item["input_audio"]["data"]
            format = item["input_audio"]["format"]
            decoded_data = base64.b64decode(data)
            with tempfile.NamedTemporaryFile(suffix=f'.{format}') as temp_file:
                temp_file.write(decoded_data)
                temp_file.flush()
                audio, sample_rate = sf.read(temp_file.name)
                # Currently Gemma3 doesn't support audio in official release
                logger.warning("Audio support is not currently implemented for Gemma3")
                
        elif item["type"] == "audio_url":
            url = item["audio_url"]["url"]
            with tempfile.NamedTemporaryFile() as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.flush()
                audio, sample_rate = sf.read(temp_file.name)
                # Currently Gemma3 doesn't support audio in official release
                logger.warning("Audio support is not currently implemented for Gemma3")
                
    return messages

def generate_response(messages, system_prompt=None, params=None):
    if model is None or processor is None:
        return "Model not loaded. Please check logs for errors."
    
    try:
        # Use provided parameters or defaults
        generation_params = {
            "max_new_tokens": params.get("max_completion_tokens", 1000),
            "do_sample": True,
            "temperature": params.get("temperature", 1.0),
            "top_k": params.get("top_k", 64),
            "top_p": params.get("top_p", 0.95),
            "min_p": params.get("min_p", 0.01),
            "repetition_penalty": params.get("repetition_penalty", 1.0)
        }
        
        logger.info(f"Generation parameters: {generation_params}")
        
        # Format messages for the model
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Add user message
        formatted_messages.append({
            "role": "user",
            "content": messages
        })
        
        # Process the messages with the Gemma3 processor
        inputs = processor.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        # Keep track of the input length to extract only the new tokens
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate the response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                **generation_params
            )
            # Extract only the newly generated tokens
            generation = generation[0][input_len:]
        
        # Decode the response
        decoded = processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Generated response: {decoded[:100]}...")  # Log first 100 chars
        
        return decoded
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating response: {str(e)}"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_completion_tokens: Optional[int] = 1000
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 64
    min_p: Optional[float] = 0.01
    repetition_penalty: Optional[float] = 1.0

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions API requests."""
    try:
        # Extract messages
        system_message = None
        user_message = None
        
        for message in request.messages:
            if message["role"] == "system":
                system_message = message
            elif message["role"] == "user":
                user_message = message
        
        if not user_message:
            logger.warning("Received request without user message")
            return JSONResponse(
                status_code=400, 
                content={"error": "No user message found in the request"}
            )
        
        # Get system prompt if available
        system_prompt = None
        if system_message and "content" in system_message:
            if isinstance(system_message["content"], list) and len(system_message["content"]) > 0:
                for item in system_message["content"]:
                    if item.get("type") == "text":
                        system_prompt = item.get("text", "")
                        break
            elif isinstance(system_message["content"], str):
                system_prompt = system_message["content"]
        
        # Process user message content
        content_list = user_message["content"]
        processed_messages = process_message_content(content_list)
        
        # Extract parameters for generation
        generation_params = {
            "max_completion_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "repetition_penalty": request.repetition_penalty
        }
        
        # Generate response
        response_text = generate_response(processed_messages, system_prompt, generation_params)
        
        # Format the response
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": MODEL_ID
        }
        
        return response
    
    except Exception as e:
        # Log the detailed error with traceback
        logger.error(f"Error processing chat completion: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        
        # Return a structured error response
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "request_id": id(request)
                }
            }
        )

@app.get("/health")
async def health():
    if model is None or processor is None:
        return {"status": "ERROR", "message": "Model not loaded"}
    return {"status": "OK", "model": MODEL_ID}