import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


HARDCODED_DASHSCOPE_API_KEY = "sk-4f0c4cc8a9c446bf82095ee86ec0af4b"


@dataclass
class StrongModelConfig:
    model: str
    api_base: str
    api_key: Optional[str] = None
    api_path: str = "/chat/completions"
    timeout: int = 120
    temperature: float = 0.3
    top_p: float = 0.8
    max_tokens: int = 512
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    sample_id: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None


@dataclass
class GenerationResult:
    sample_id: str
    text: str
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    raw_response: Dict[str, Any]
    error: Optional[str] = None


class StrongModelClient:
    """Thin wrapper around an OpenAI-compatible chat completion API."""

    def __init__(self, config: StrongModelConfig):
        self.config = config

    @classmethod
    def from_args(cls, args) -> "StrongModelClient":
        extra_body: Dict[str, Any] = {}
        if getattr(args, "llm_extra_body", None):
            extra_body = json.loads(args.llm_extra_body)
        # Force-disable thinking mode for distillation runs.
        extra_body["enable_thinking"] = False

        config = StrongModelConfig(
            model=args.llm_model,
            api_base=args.llm_api_base.rstrip("/"),
            api_key=getattr(args, "llm_api_key", None) or HARDCODED_DASHSCOPE_API_KEY,
            api_path=args.llm_api_path,
            timeout=args.llm_timeout,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_tokens=args.llm_max_tokens,
            presence_penalty=args.llm_presence_penalty,
            frequency_penalty=args.llm_frequency_penalty,
            seed=args.llm_seed,
            extra_body=extra_body,
        )
        return cls(config)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _url(self) -> str:
        return f"{self.config.api_base}{self.config.api_path}"

    def _format_error(self, exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            response = exc.response
            detail = response.text.strip()
            try:
                detail = json.dumps(response.json(), ensure_ascii=False)
            except ValueError:
                pass
            if detail:
                return f"HTTPError {response.status_code}: {detail}"
            return f"HTTPError {response.status_code}: {exc}"
        return f"{type(exc).__name__}: {exc}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": request.messages,
            "temperature": self.config.temperature if request.temperature is None else request.temperature,
            "top_p": self.config.top_p if request.top_p is None else request.top_p,
            "max_tokens": self.config.max_tokens if request.max_tokens is None else request.max_tokens,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if request.stop:
            payload["stop"] = request.stop
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        try:
            response = requests.post(
                self._url(),
                headers=self._headers(),
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content", "")
            return GenerationResult(
                sample_id=request.sample_id,
                text=text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage"),
                raw_response=data,
            )
        except Exception as exc:
            return GenerationResult(
                sample_id=request.sample_id,
                text="",
                finish_reason=None,
                usage=None,
                raw_response={},
                error=self._format_error(exc),
            )

    def generate_batch(
        self,
        requests_list: List[GenerationRequest],
        max_concurrency: int = 8,
    ) -> List[GenerationResult]:
        if not requests_list:
            return []

        worker_num = max(1, min(max_concurrency, len(requests_list)))
        with ThreadPoolExecutor(max_workers=worker_num) as executor:
            results = list(executor.map(self.generate, requests_list))
        return results


def is_likely_balance_error(error_message: Optional[str]) -> bool:
    if not error_message:
        return False

    normalized_error = error_message.lower()
    keywords = [
        "insufficient",
        "balance",
        "arrear",
        "quota",
        "payment",
        "bill",
        "余额",
        "欠费",
        "充值",
        "账户余额",
    ]
    return any(keyword in normalized_error for keyword in keywords)
