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

answer = llm.generate("Explain fast inference in simple words")
print(answer)
