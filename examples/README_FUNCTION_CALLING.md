# Function Calling in exo

Function calling (or tool calling) is a powerful capability that allows large language models to invoke external functions when they need specific information to answer a user query. This README explains how to implement and use function calling with the exo ChatGPT-compatible API.

## Overview

Function calling in exo works similarly to OpenAI's API:

1. Your application sends a request with tool definitions
2. The model may generate text that includes function call markup
3. Your application parses the response to extract function calls
4. Your application executes the corresponding functions
5. Your application sends the function results back to the model for a final answer

## API Format

When making a request to the exo API with function calling, you should format your request like this:

```json
{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "What's the weather in Boston?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto" 
}
```

The `tool_choice` parameter can be:
- `"auto"`: Let the model decide when to use tools
- `"none"`: Never use tools
- `{"type": "function", "function": {"name": "function_name"}}`: Force the model to use a specific function

## Model Response Format

When a model decides to use a function, the response will include special markup indicating the function call:

```
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston", "unit": "celsius"}}
</tool_call>
```

Your application needs to parse this format to extract and execute the function calls.

## Example Implementation

For a full implementation example, see `function_calling.py` in this directory. The example shows:

1. How to format requests with tool definitions
2. How to parse responses to extract function calls
3. How to execute functions and send results back to the model

## Supporting Models

Function calling works best with models that have been fine-tuned for tool use, such as:
- Llama 3.1
- Mistral
- Claude
- Qwen
- DeepSeek
- Phi-3

Note that not all models perform tool calling equally well. Some may require specific instruction prompting to use tools effectively.

## Tips and Best Practices

1. **Clear Function Descriptions**: Write clear, concise descriptions for your functions and parameters
2. **Error Handling**: Implement robust error handling for function parsing and execution
3. **Model Selection**: Choose models known to work well with function calling
4. **Testing**: Test your implementation with various queries to ensure reliable function calling

For more information and advanced usage, refer to the exo documentation or contact the exo team.