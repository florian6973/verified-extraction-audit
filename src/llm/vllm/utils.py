import warnings
import numpy as np
import requests
import json
from pprint import pprint

from src._repo import REPO_ROOT
def call_vllm_server(prompt, model="Qwen3-32B-AWQ", stream=False, repetition_penalty=1.0, port=12346, task="generate_kg", note="<unspecified>",
                     temperature=None, top_p=None, 
                     top_k=None, min_p=None, max_tokens=32768, add_chat_template=True,
                     perplexity=False):
    """Call the vLLM server for text generation.
    
    Args:
        prompt (str): The input prompt
        model_name (str): Name of the model (not used in vLLM server)
        stream (bool): Whether to use streaming response
    
    Returns:
        str: The generated text or None if there was an error
    """
    if add_chat_template:
        url = f"http://localhost:{port}/v1/chat/completions" # chat completions vs completions
    else:
        url = f"http://localhost:{port}/v1/completions" # chat completions vs completions
    # or v1/completions

    if perplexity:
        assert stream == False, "Perplexity is not supported for streaming"
        assert add_chat_template == False, "Perplexity is not supported for chat completions"
    

    payload = {
        "max_tokens": max_tokens,
        # "temperature": 0.0,
        # "top_p": 1.0,
        "repetition_penalty": repetition_penalty, # 1.2
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
        # "extra_body": {
        #     "top_k": 50,                       # match HF
        #     "spaces_between_special_tokens": False,
        #     "skip_special_tokens": False
        # }
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
        if top_k is not None:
            payload['top_k'] = top_k
        else:
            payload['top_k'] = 50
        if min_p is not None:
            payload['min_p'] = min_p
        # else:
            # payload['min_p'] = 0.0
        payload['skip_special_tokens'] = False
        payload['spaces_between_special_tokens'] = False
    if perplexity:
        # payload['logprobs'] = 1
        payload['echo'] = True
        payload['prompt_logprobs'] = 1

    if add_chat_template:
        payload['messages'] = [{"role": "user", "content": prompt}]
    else:
        payload['prompt'] = prompt
    
    try:
        if stream:
            # For streaming, we need to handle the SSE response
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            print(response)
            
            full_text = ""
            for line in response.iter_lines():
                # print(line)
                if line:
                    # Remove "data: " prefix and parse JSON
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]  # Remove "data: " prefix
                        if line == "[DONE]":
                            break
                        # print(line)
                        try:
                            data = json.loads(line)
                            # print(data)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("text", "")
                                if not delta:
                                    delta = data["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    full_text += delta
                                    print(delta, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
            
            print()  # Add newline at the end
            return full_text
        else:
            # Non-streaming response
            response = requests.post(url, json=payload)
            response.raise_for_status()
            # pprint(response.json()['choices'])
            
            # print(tokens)
            if perplexity:
                tokens = []
                for log_probs in response.json()['choices'][0]['prompt_logprobs']:
                    if log_probs is not None:
                        print(log_probs)
                        tokens.append(log_probs[list(log_probs.keys())[0]]['logprob'])
                # tokens = torch.tensor(tokens)
                # print(tokens)
                tokens = - sum(tokens[1:-1]) / len(tokens[1:-1]) # equivalent to model loss
                warnings.warn("Make sure to add <|begin_of_text|> and <|end_of_text|> to the prompt")
                return np.exp(tokens)
            # print(np.exp(tokens))
            # print(tokens)
            if add_chat_template:
                return response.json()["choices"][0].get("message", {}).get("content", "")
            else:
                return response.json()["choices"][0].get("text", "")
    except Exception as e:
        print(f"Error calling vLLM server: {str(e)}")
        return None
    
if __name__ == "__main__":
    print("Perplexity")
    prompt = "Life is too short to be wasted on"
    # add special beginning token
    prompt = "<|begin_of_text|>" + prompt + "<|end_of_text|>"
    response = call_vllm_server(prompt, temperature=0.7, top_p=0.8, max_tokens=0, stream=False, repetition_penalty=1.2, perplexity=True, add_chat_template=False)
    pprint(response)

    # # load huggingface model
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # tokenizer = AutoTokenizer.from_pretrained(" + REPO_ROOT + "/outputs_models/finetuning/Llama_3.2-1B-1-0.1-False-1/Llama_3.2-1B")
    # model = AutoModelForCausalLM.from_pretrained(" + REPO_ROOT + "/outputs_models/finetuning/Llama_3.2-1B-1-0.1-False-1/Llama_3.2-1B")
    

    # prompts = ['Generate', 'Generate a', 'Generate a clinical', 'Generate a clinical note']
    # for prompt in prompts:
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     loss = model(**inputs, labels=inputs['input_ids']).loss
    #     print(prompt, loss)

    
    
    
    