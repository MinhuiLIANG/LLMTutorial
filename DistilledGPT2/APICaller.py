import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Hello GPT"}
)
print(response.json())