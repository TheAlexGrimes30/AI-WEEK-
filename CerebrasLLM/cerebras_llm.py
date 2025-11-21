import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()


class CerebrasLLM:
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b"):
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.model = model
        self.client = Cerebras(api_key=self.api_key)

    def generate(self, prompt: str,
                 max_tokens: int = 2000,
                 temperature: float = 0.6,
                 top_p: float = 1.0) -> str:

        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )

        return completion.choices[0].message.content

llm = CerebrasLLM()

prompt = (
        "Ты — ML-инженер с опытом 10 лет и опытный предприниматель.\n"
        f"Сгенерируй ровно 10 кратких идей AI-проектов на русском языке.\n"
        "Каждая идея — 2–3 предложения.\n"
        "Формат вывода строго такой:\n"
        "ИДЕЯ 1: <текст>\n"
        "ИДЕЯ 2: <текст>\n"
        "... до нужного количества.\n\n"
        "Важно:\n"
        "- не использовать списки, маркеры, тире\n"
        "- одна идея = одна строка\n"
        "- все строки должны начинаться с 'ИДЕЯ X:'\n"
        "- не пропускай номера"
    )

answer = llm.generate(prompt)
print(answer)
