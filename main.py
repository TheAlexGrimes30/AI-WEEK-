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
        {"role": "system", "text": "Ты — эксперт по генерации AI-проектов."},
        {"role": "user", "text": "Сгенерируй 8–10 кратких идей AI-проектов. Каждая — 2–3 предложения. "
                                 "Формат: список строк, по одному проекту в строке."}
    ]
    idea_result = client.generate(idea_prompt)
    raw_text = idea_result["result"]["alternatives"][0]["message"]["text"]
    ai_ideas = [line.lstrip("-• ").strip() for line in raw_text.split("\n") if line.strip()]

    improved_prompt = (
        "Ты — опытный ML-архитектор и инженер. "
        "На вход приходит идея AI-проекта. Сгенерируй развернутое техническое описание, "
        "список необходимых технологий и библиотек, основные этапы реализации (3–5 пунктов) "
        "и оценку сложности (легко/средне/сложно). "
        "Выводи каждый раздел с заголовком для удобного чтения:\n\n"
        "Техническое описание:\n<текст>\n\n"
        "Необходимые технологии и библиотеки:\n- технология 1\n- технология 2\n...\n\n"
        "Этапы реализации:\n1. этап 1\n2. этап 2\n...\n\n"
        "Оценка сложности:\n<легко/средне/сложно>\n\n"
        "Старайся давать конкретные, логичные и практичные рекомендации."
    )

    for idx, idea in enumerate(ai_ideas, start=1):
        messages = [
            {"role": "system", "text": improved_prompt},
            {"role": "user", "text": idea}
        ]
        resp = client.generate(messages, temperature=0.4, max_tokens=3000)
        text = resp["result"]["alternatives"][0]["message"]["text"]

        print(f"\n{'='*60}")
        print(f"Проект #{idx}")
        print(f"Идея: {idea}\n")
        sections = ["Техническое описание:", "Необходимые технологии и библиотеки:",
                    "Этапы реализации:", "Оценка сложности:"]
        start = 0
        for i, header in enumerate(sections):
            pos = text.find(header, start)
            if pos == -1:
                continue
            next_pos = len(text)
            if i + 1 < len(sections):
                next_pos = text.find(sections[i+1], pos)
                if next_pos == -1:
                    next_pos = len(text)
            content = text[pos + len(header):next_pos].strip()
            print(f"{header}\n{content}\n")
            start = next_pos
        print("="*60)