const fs = await import('fs');

const response = await fetch("http://localhost:52415/v1/chat/completions", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        "model": "llama-3.2-1b",
        "messages": [
            {
                "role": "user",
                "content": "Please call the get_weather function to determine the weather in Beijing"
            }
        ],
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    })
});

console.log(response.status)
console.log(await response.text());
