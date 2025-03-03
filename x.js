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
                "content": "Please call the get_weather function (signature get_weather(city: string) to determine the weather in Beijing"
            }
        ],
        "response_format": {
            "type": "lark_grammar",
            "lark_grammar": fs.readFileSync("./g.lark", "utf8")
        }
    })
});

console.log(response.status)
console.log(await response.text());
