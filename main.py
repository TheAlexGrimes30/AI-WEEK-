import os
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

class YandexLLM:
    def __init__(self, yandex_api_key: str, yandex_model_uri: str = os.getenv("YANDEX_URI")):
        self.yandex_api_key = yandex_api_key
        self.yandex_model_uri = yandex_model_uri
        self.yandex_url = os.getenv("YANDEX_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.yandex_api_key}"
        }

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.6,
                 max_tokens: int = 2000, stream: bool = False) -> Dict[str, Any]:
        payload = {
            "modelUri": self.yandex_model_uri,
            "completionOptions": {
                "stream": stream,
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "messages": messages
        }

        response = requests.post(self.yandex_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    yandex_api_key = os.getenv("YANDEX_API_KEY")
    client = YandexLLM(yandex_api_key)

    idea_prompt = [
        {"role": "system", "text": "Сгенерируй 8–10 кратких идей AI-проектов. Каждая — 2–3 предложения. Формат: список строк, по одному проекту в строке."},
        {"role": "user", "text": "Сгенерируй идеи."}
    ]

    idea_result = client.generate(idea_prompt)

    raw_text = idea_result["result"]["alternatives"][0]["message"]["text"]

    ai_ideas = [
        line.lstrip("-• ").strip()
        for line in raw_text.split("\n")
        if line.strip()
    ]

    messages = [
        {"role": "system", "text": "Ты — ML-архитектор. На вход приходит идея проекта, а ты генерируешь: развернутое техописание, список технологий, этапы реализации, оценку сложности."}
    ]

    results = []
    for idea in ai_ideas:
        messages.append({"role": "user", "text": idea})
        resp = client.generate(messages)
        results.append({"idea": idea, "response": resp})
        messages.pop()

    for item in results:
        print("==== Проект ====")
        print(item["idea"])
        print("==== Ответ LLM ====")
        print(item["response"]["result"]["alternatives"][0]["message"]["text"])
        print()
