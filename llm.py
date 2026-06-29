from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read config from environment first, then Streamlit secrets when deployed."""
    val = os.getenv(name)
    if val is not None:
        return val
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


class QwenClient:
    def __init__(self):
        self.backend = (get_setting("LLM_BACKEND", "mock") or "mock").lower()
        self.model_id = get_setting("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        self._pipe = None

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1200) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        if self.backend == "openrouter":
            return self._generate_openrouter(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
        if self.backend == "groq":
            return self._generate_groq(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
        if self.backend == "mock":
            return self._generate_mock(user_prompt)
        return self._generate_transformers(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    def _generate_transformers(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        if self._pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._pipe.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)[0]["generated_text"]
        return out[len(prompt):].strip()

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        model = get_setting("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=240)
        res.raise_for_status()
        return res.json()["message"]["content"]

    def _generate_openrouter(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = get_setting("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in environment variables or Streamlit secrets.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        }
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def _generate_groq(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        api_key = os.getenv("GROQ_API_KEY", "")
        model = get_setting("GROQ_MODEL", "llama-3.1-8b-instant")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Set it in environment variables or Streamlit secrets.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        }
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def _generate_mock(self, user_prompt: str) -> str:
        chunk_ids = re.findall(r"CHUNK_ID:\s*(C\d+)", user_prompt)
        query = _extract_between(user_prompt, "USER QUESTION:", "RETRIEVED CHUNKS:").strip()
        if "novel" in user_prompt.lower() or "NOV-001" in user_prompt:
            return json.dumps({
                "user_question": query,
                "answer": {
                    "text": "Mock mode: retrieved chunks were processed. Switch to an LLM backend for real novelty analysis.",
                    "confidence": "low",
                    "answer_chunk_ids": chunk_ids[:2],
                    "limitations": "Mock mode does not perform real reasoning."
                },
                "novel_insights": [],
                "message": "No novel insights found",
            })
        return json.dumps({
            "user_question": query,
            "answer": {
                "text": "Mock mode: retrieved chunks were processed. Switch to an LLM backend for real discrepancy analysis.",
                "confidence": "low",
                "answer_chunk_ids": chunk_ids[:2],
                "limitations": "Mock mode does not perform real reasoning."
            },
            "discrepancies": [],
            "message": "No discrepancies found",
        })


def _extract_between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except Exception:
        return ""


def parse_json_response(text: str, fallback_kind: str = "discrepancy") -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    if fallback_kind == "novelty":
        return {
            "user_question": "",
            "answer": {"text": "Model output could not be parsed as JSON.", "confidence": "low", "answer_chunk_ids": [], "limitations": "JSON parsing failed."},
            "novel_insights": [
                {
                    "id": "NOV-PARSE",
                    "strength": "HIGH",
                    "category": "Model Output Parsing",
                    "risk_type": "Technical / Implementation Risk",
                    "chunk_ids": [],
                    "title": "Model did not return valid JSON",
                    "insight": "The model response could not be parsed as JSON. The prompt, chunk count, or max token setting may need adjustment.",
                    "evidence": [{"chunk_id": "N/A", "source": "Model output", "quote_or_phrase": text[:500]}],
                    "assumptions_made": ["The model response is assumed to contain useful information, but it could not be parsed into the expected JSON schema."],
                    "why_it_matters": "The UI needs structured JSON to display novelty insights reliably.",
                    "suggested_follow_up": "Retry with fewer chunks or a stricter JSON-output model.",
                }
            ],
            "message": "Novel insights found",
        }

    return {
        "user_question": "",
        "answer": {"text": "Model output could not be parsed as JSON.", "confidence": "low", "answer_chunk_ids": [], "limitations": "JSON parsing failed."},
        "discrepancies": [
            {
                "id": "INC-PARSE",
                "severity": "HIGH",
                "type": "Other",
                "chunk_ids": [],
                "title": "Model did not return valid JSON",
                "description": "The model response could not be parsed as JSON. The prompt, chunk count, or max token setting may need adjustment.",
                "evidence": [{"chunk_id": "N/A", "source": "Model output", "quote_or_phrase": text[:500]}],
                "why_it_matters": "The UI needs structured JSON to display discrepancies reliably.",
                "suggested_resolution": "Retry with fewer chunks or a stricter JSON-output model.",
            }
        ],
        "message": "Discrepancies found",
    }


def normalize_discrepancy_payload(data: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    order = {"HIGH": 0, "High": 0, "MEDIUM": 1, "Medium": 1, "LOW": 2, "Low": 2}
    discrepancies = data.get("discrepancies") or []
    for d in discrepancies:
        d.setdefault("risk_type", "Other")
        d.setdefault("assumptions_made", [])
    discrepancies = sorted(discrepancies, key=lambda d: order.get(d.get("severity", "LOW"), 3))[:limit]
    data["discrepancies"] = discrepancies
    data["discrepancies_found"] = bool(discrepancies)
    data["message"] = "Discrepancies found" if discrepancies else "No discrepancies found"
    return data


def normalize_novelty_payload(data: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    order = {"HIGH": 0, "High": 0, "MEDIUM": 1, "Medium": 1, "LOW": 2, "Low": 2}
    insights = data.get("novel_insights") or []
    for d in insights:
        d.setdefault("risk_type", "Other")
        d.setdefault("assumptions_made", [])
    insights = sorted(insights, key=lambda d: order.get(d.get("strength", "LOW"), 3))[:limit]
    data["novel_insights"] = insights
    data["novel_insights_found"] = bool(insights)
    data["message"] = "Novel insights found" if insights else "No novel insights found"
    return data
