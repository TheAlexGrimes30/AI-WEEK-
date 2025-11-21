import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import re

load_dotenv()

class CerebrasLLM:
    """
    Класс для работы с LLM от Cerebras через облачный API.

    Attributes:
        api_key (str): API-ключ для доступа к Cerebras.
        model (str): Используемая модель LLM.
        client (Cerebras): Клиент Cerebras SDK для взаимодействия с API.
    """

    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b"):
        """
        Инициализация клиента Cerebras LLM.

        Args:
            api_key (str, optional): API-ключ. Если не указан, берется из переменных окружения.
            model (str, optional): Модель для генерации текста. Defaults to "llama-3.3-70b".
        """

        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.model = model
        self.client = Cerebras(api_key=self.api_key)

    def generate(self, prompt: str,
                 max_tokens: int = 8000,
                 temperature: float = 0.4,
                 top_p: float = 1.0,
                 stream: bool = True) -> str:
        """
        Генерация текста с помощью модели Cerebras.

        Args:
            prompt (str): Текст запроса для модели.
            max_tokens (int, optional): Максимальное количество токенов в ответе. Defaults to 8000.
            temperature (float, optional): Параметр креативности генерации. Defaults to 0.4.
            top_p (float, optional): Параметр сэмплирования. Defaults to 1.0.
            stream (bool, optional): Если True, возвращает текст по частям (streaming). Defaults to True.

        Returns:
            str: Сгенерированный текст.
        """

        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream
        )

        if stream:
            text = ""
            for chunk in completion:
                delta_content = getattr(getattr(chunk.choices[0], "delta", None), "content", None)
                if delta_content:
                    text += delta_content
            return text
        else:
            return completion.choices[0].message.content

def pretty_print_projects(text: str) -> None:
    """
    Красивый вывод проектов с разделами, учитывая только нумерацию от 1 до 10.

    Args:
        text (str): Текст с идеями и техническими описаниями проектов.

    Returns:
        None
    """

    sections = [
        "Техническое описание:",
        "Необходимые технологии и библиотеки:",
        "Этапы реализации:",
        "Оценка сложности:"
    ]

    matches = re.finditer(r"ИДЕЯ (\d+):", text)
    project_positions = [(int(m.group(1)), m.start()) for m in matches]

    project_positions = [p for p in project_positions if 1 <= p[0] <= 10]

    for i, (proj_num, start_pos) in enumerate(project_positions):
        end_pos = project_positions[i + 1][1] if i + 1 < len(project_positions) else len(text)
        proj_text = text[start_pos:end_pos].strip()

        if not proj_text or proj_text == f"ИДЕЯ {proj_num}:":
            continue

        print("\n" + "="*70)
        print(f"Проект #{proj_num}")
        print("="*70 + "\n")

        start = 0
        for j, header in enumerate(sections):
            pos = proj_text.find(header, start)
            if pos == -1:
                continue
            next_pos = len(proj_text)
            if j + 1 < len(sections):
                next_pos = proj_text.find(sections[j+1], pos)
                if next_pos == -1:
                    next_pos = len(proj_text)
            content = proj_text[pos + len(header):next_pos].strip()
            if content:
                print(f"{header}\n{content}\n")
            start = next_pos


if __name__ == "__main__":
    """
    Пример использования CerebrasLLM для генерации 10 AI-проектов
    с развернутым техническим описанием и выводом в консоль.
    """

    llm = CerebrasLLM()

    prompt = (
        "Ты — ML-инженер с 10-летним опытом и опытный предприниматель.\n"
        "Тебе необходимо выполнить две связанные задачи:\n\n"
        "1) Сгенерировать ровно 10 кратких идей AI-проектов на русском языке.\n"
        "Каждая идея — 2–3 предложения.\n"
        "Формат строго такой:\n"
        "ИДЕЯ 1: <текст>\n"
        "ИДЕЯ 2: <текст>\n"
        "ИДЕЯ 3: <текст>\n"
        "ИДЕЯ 4: <текст>\n"
        "ИДЕЯ 5: <текст>\n"
        "ИДЕЯ 6: <текст>\n"
        "ИДЕЯ 7: <текст>\n"
        "ИДЕЯ 8: <текст>\n"
        "ИДЕЯ 9: <текст>\n"
        "ИДЕЯ 10: <текст>\n\n"
        "2) Для каждой идеи создай развернутое техническое описание проекта с разделами:\n"
        "Техническое описание:\n<текст>\n\n"
        "Необходимые технологии и библиотеки:\n- технология 1\n- технология 2\n...\n\n"
        "Этапы реализации:\n1. этап 1\n2. этап 2\n...\n\n"
        "Оценка сложности:\n<легко/средне/сложно>\n\n"
        "Требования:\n"
        "- описание должно быть техническим, реалистичным и подробным\n"
        "- этапов реализации должно быть 3–5\n"
        "- писать строго на русском\n"
        "- сохранять структуру разделов без изменений\n\n"
        "Сначала выведи 10 идей, затем последовательно развернутое описание каждого проекта."
    )

    answer = llm.generate(prompt)
    pretty_print_projects(answer)
