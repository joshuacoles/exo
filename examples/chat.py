#!/usr/bin/env python3
"""
Simple AI chat REPL with improved user experience.
Allows interactive conversation with an AI model through the command line.
Supports streaming responses from the AI model.
Includes readline support for command history within the session.
"""

import json
import readline
import requests
import sys
import time
from typing import List, Dict, Iterator

def chat_completion(messages: List[Dict[str, str]], model: str = "llama-3.2-1b", stream: bool = False) -> dict:
    """Send chat completion request to local exo server."""
    response = requests.post(
        "http://localhost:6300/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "stream": stream
        },
        stream=stream
    )

    if not stream:
        return response.json()
    else:
        return response


def process_stream(response) -> Iterator[str]:
    """Process streaming response and yield content chunks."""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            # Skip the "data: " prefix
            if line.startswith('data: '):
                line = line[6:]
                # Skip [DONE] message
                if line == '[DONE]':
                    break
                try:
                    data = json.loads(line)
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


def print_help():
    """Display available commands."""
    print("\n\033[1mAvailable Commands:\033[0m")
    print("  \033[94m/help\033[0m    - Show this help message")
    print("  \033[94m/clear\033[0m   - Clear the conversation history")
    print("  \033[94m/model\033[0m   - Show or change the current model")
    print("  \033[94m/history\033[0m - Show conversation history")
    print("  \033[94m/exit\033[0m    - Exit the chat")


def handle_command(command: str, messages: List[Dict[str, str]], model: str) -> tuple[bool, str]:
    """Handle special commands starting with /."""
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        print_help()
        return True, model

    elif cmd == "/clear":
        return True, model

    elif cmd == "/model":
        if len(parts) > 1:
            new_model = parts[1]
            print(f"\033[92mSwitched model to: {new_model}\033[0m")
            return True, new_model
        else:
            print(f"\033[92mCurrent model: {model}\033[0m")
            return True, model

    elif cmd == "/history":
        if not messages:
            print("\033[93mNo conversation history yet.\033[0m")
        else:
            print("\n\033[1mConversation History:\033[0m")
            for i, msg in enumerate(messages):
                role = msg["role"].capitalize()
                content = msg["content"]
                role_color = "\033[94m" if role == "User" else "\033[92m"
                print(f"{role_color}{role}:\033[0m {content}")
        return True, model

    elif cmd == "/exit":
        print("\n\033[92mGoodbye!\033[0m")
        sys.exit(0)

    return False, model


def format_user_prompt():
    """Format the user prompt with color."""
    return "\033[94mYou:\033[0m "


def format_ai_prompt():
    """Format the AI prompt with color."""
    return "\033[92mAI:\033[0m "


def main():
    # Initialize conversation history and model
    messages = []
    model = "llama-3.2-1b"

    # Print welcome message
    print("\033[1mAI Chat REPL\033[0m")
    print("Type \033[94m/help\033[0m for available commands")
    print("Press Ctrl+C to clear current input, Ctrl+D or type \033[94m/exit\033[0m to quit")

    try:
        while True:
            # Get user input with handling for Ctrl+C and Ctrl+D
            try:
                user_input = input(format_user_prompt()).strip()
            except KeyboardInterrupt:
                # Ctrl+C: clear current input line
                print("\n\033[93mInput cleared\033[0m")
                continue
            except EOFError:
                # Ctrl+D: exit gracefully
                print("\033[92mGoodbye!\033[0m")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Check for commands
            if user_input.startswith('/'):
                is_command, model = handle_command(user_input, messages, model)
                if is_command:
                    if user_input.lower() == "/clear":
                        messages = []
                        print("\033[93mConversation history cleared.\033[0m")
                    continue

            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Get AI response
            try:
                print(format_ai_prompt(), end="", flush=True)

                # Use streaming response
                start_time = time.time()
                response = chat_completion(messages, model=model, stream=True)
                full_content = ""

                for content_chunk in process_stream(response):
                    full_content += content_chunk
                    print(content_chunk, end="", flush=True)

                # Add complete message to history
                messages.append({
                    "role": "assistant",
                    "content": full_content
                })

                # Show response time for longer responses
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    print(f"\n\033[90m(Response time: {elapsed:.2f}s)\033[0m")
                else:
                    print()  # Add a newline after response

            except Exception as e:
                print(f"\n\033[91mError: {str(e)}\033[0m")
                # Remove failed message from history
                messages.pop()

    except KeyboardInterrupt:
        # This should not be reached with the inner try/except, but just in case
        print("\n\033[92mGoodbye!\033[0m")


if __name__ == "__main__":
    main()
