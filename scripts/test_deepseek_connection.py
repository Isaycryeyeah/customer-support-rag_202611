import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from openai import OpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def main() -> None:
    if not DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Check your .env file.")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=30.0)

    start = time.time()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": "请用一句话介绍你自己。"}],
        temperature=0.2,
        max_tokens=100,
    )
    latency_ms = (time.time() - start) * 1000

    print(f"model: {DEEPSEEK_MODEL}")
    print(f"latency: {latency_ms:.1f} ms")
    print(f"reply: {response.choices[0].message.content}")
    print(f"usage: {response.usage}")


if __name__ == "__main__":
    main()
