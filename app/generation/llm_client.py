"""Bounded generation with explicit failure and usage metadata."""
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AuthenticationError, OpenAI, RateLimitError)
from app.generation.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
FALLBACK_ANSWER = """结论：
当前回答服务暂时不可用，请稍后重试或联系人工客服。

处理步骤：
1. 请保留设备型号和问题描述，稍后重新提交。
2. 如问题紧急，请联系官方售后渠道。

注意事项：
- 这是服务故障提示，不能作为售后问题的处理结论。

引用来源：
- 无
"""


@dataclass
class GenerationResult:
    success: bool
    text: str = ''
    provider: str = ''
    model: str = ''
    error_type: str | None = None
    attempts: int = 0
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    thinking: str = ''

    def to_dict(self):
        return asdict(self)


def error_category(error):
    if isinstance(error, APITimeoutError):
        return 'timeout'
    if isinstance(error, RateLimitError):
        return 'rate_limit'
    if isinstance(error, AuthenticationError):
        return 'authentication'
    if isinstance(error, APIConnectionError):
        return 'connection'
    if isinstance(error, APIStatusError):
        return 'upstream_error'
    return 'internal_error'


class LLMClient:
    def __init__(self, provider='deepseek', *, client=None, timeout=None,
                 budget=None, max_attempts=None):
        self.provider = provider
        self.timeout = float(timeout if timeout is not None else os.getenv('LLM_TIMEOUT_SECONDS', '12'))
        self.budget = float(budget if budget is not None else os.getenv('LLM_BUDGET_SECONDS', '30'))
        self.max_attempts = int(max_attempts if max_attempts is not None else os.getenv('LLM_MAX_ATTEMPTS', '2'))
        if self.timeout <= 0 or self.budget <= 0 or self.max_attempts < 1:
            raise ValueError('Generation timeout/budget must be positive; attempts must be >= 1.')
        self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat') if provider == 'deepseek' else provider
        self.thinking = os.getenv('DEEPSEEK_THINKING', 'disabled')
        if self.thinking not in ('enabled', 'disabled'):
            raise ValueError('DEEPSEEK_THINKING must be enabled or disabled.')
        if provider == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if client is None and not api_key:
                raise ValueError('DEEPSEEK_API_KEY is not set.')
            self.client = client if client is not None else OpenAI(
                api_key=api_key, base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                timeout=self.timeout, max_retries=0)
        elif provider != 'mock':
            raise ValueError(f'Unsupported provider: {provider}')

    def generate_safe(self, prompt, temperature=0.2):
        start = time.perf_counter()
        result = GenerationResult(False, provider=self.provider, model=self.model, thinking=self.thinking)
        if self.provider == 'mock':
            result.success = True
            result.text = '结论：测试替身回答。\n处理步骤：联系官方售后。\n注意事项：以检测结果为准。'
            result.finish_reason = 'stop'
        else:
            for attempt in range(1, self.max_attempts + 1):
                remaining = self.budget - (time.perf_counter() - start)
                if remaining <= 0:
                    result.error_type = 'timeout'
                    break
                result.attempts = attempt
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                                                    {'role': 'user', 'content': prompt}],
                        temperature=temperature, max_tokens=1024,
                        extra_body={'thinking': {'type': self.thinking}},
                        timeout=min(self.timeout, remaining))
                    result.usage = response.usage.model_dump() if response.usage else {}
                    choice = response.choices[0] if response.choices else None
                    result.finish_reason = choice.finish_reason if choice else None
                    text = choice.message.content if choice else None
                    if result.finish_reason == 'length':
                        result.error_type = 'truncated_output'
                    elif not isinstance(text, str) or not text.strip():
                        result.error_type = 'empty_output'
                    else:
                        result.success, result.text, result.error_type = True, text.strip(), None
                    break
                except Exception as error:
                    result.error_type = error_category(error)
                    logger.warning('Generation failed: category=%s attempt=%d', result.error_type, attempt)
                    retryable = isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError))
                    if not retryable or attempt == self.max_attempts:
                        break
                    delay = min(float(attempt), max(0, self.budget - (time.perf_counter() - start)))
                    time.sleep(delay)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

    def generate(self, prompt, temperature=0.2):
        """Strict convenience entry point; never return a fallback as success."""
        result = self.generate_safe(prompt, temperature)
        if not result.success:
            raise RuntimeError(f'Generation unavailable: {result.error_type}')
        return result.text
