import os
import time
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()


class LLMBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMBase):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__(name=f"Gemini ({model})")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


MAX_RETRIES = 5

def safe_generate(model: LLMBase, prompt: str):
    for attempt in range(MAX_RETRIES):
        try:
            return model.generate(prompt)
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"{model.name} попытка {attempt+1}/{MAX_RETRIES} провалилась: {e}. Ждем {wait_time} сек.")
            time.sleep(wait_time)
    return None


def generate_projects(models: List[LLMBase], ideas: List[str]):
    improved_prompt = (
        "Ты — опытный ML-архитектор и инженер. "
        "На вход приходит идея AI-проекта. Сгенерируй развернутое техническое описание, "
        "список необходимых технологий и библиотек, основные этапы реализации (3–5 пунктов) "
        "и оценку сложности (легко/средне/сложно). "
        "Выводи каждый раздел с заголовком для удобного чтения.\n\n"
        "Техническое описание:\n<текст>\n\n"
        "Необходимые технологии и библиотеки:\n- технология 1\n- технология 2\n...\n\n"
        "Этапы реализации:\n1. этап 1\n2. этап 2\n...\n\n"
        "Оценка сложности:\n<легко/средне/сложно>\n\n"
        "Пиши строго на русском."
    )

    results = {}

    for idx, idea in enumerate(ideas, start=1):

        print("\n" + "=" * 60)
        print(f"Проект #{idx}")
        print(f"Идея: {idea}\n")

        prompt = f"{improved_prompt}\n\nИдея проекта: {idea}"

        results[idx] = {}

        for model in models:
            text = safe_generate(model, prompt)
            if text is None:
                text = "Не удалось сгенерировать ответ после нескольких попыток."
            results[idx][model.name] = text

            sections = [
                "Техническое описание:",
                "Необходимые технологии и библиотеки:",
                "Этапы реализации:",
                "Оценка сложности:"
            ]

            start = 0
            for i, header in enumerate(sections):
                pos = text.find(header, start)
                if pos == -1:
                    continue

                next_pos = len(text)
                if i + 1 < len(sections):
                    next_pos = text.find(sections[i + 1], pos)
                    if next_pos == -1:
                        next_pos = len(text)

                content = text[pos + len(header):next_pos].strip()
                print(f"{header}\n{content}\n")

                start = next_pos

    return results

def generate_ai_ideas(model: LLMBase, count: int = 10) -> List[str]:

    idea_prompt = (
        "Ты — эксперт по генерации AI-проектов.\n"
        f"Сгенерируй {count} кратких идей AI-проектов. "
        "Каждая идея — 2–3 предложения.\n"
        "Формат: список строк, по одной идее в строке.\n"
        "Не нумеровать. Не использовать маркеры типа '-' или '•'.\n"
        "Просто строки текста, одна идея — одна строка."
    )

    raw = safe_generate(model, idea_prompt)

    if raw is None:
        return []

    ideas = [line.strip() for line in raw.split("\n") if line.strip()]
    return ideas

if __name__ == "__main__":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = GeminiProvider(api_key=gemini_api_key)

    ai_ideas = generate_ai_ideas(gemini_model, count=10)

    models = [gemini_model]

    generate_projects(models, ai_ideas)