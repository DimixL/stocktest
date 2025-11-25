"""
StockSense • Streamlit MVP
==========================

Как включить внешние провайдеры:
- Переменные окружения:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o-mini"
    export FINNHUB_API_KEY="..."
    export OLLAMA_URL="http://localhost:11434"  # можно указать и http://localhost:11434/api/generate
    export OLLAMA_MODEL="mistral"
- ~/.streamlit/secrets.toml пример:
    OPENAI_API_KEY = "sk-..."
    OPENAI_MODEL = "gpt-4o-mini"
    FINNHUB_API_KEY = "..."
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "mistral"
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import requests_cache
import streamlit as st
from textwrap import dedent

# -----------------------------------------------------------------------------
# Streamlit & caching configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")
requests_cache.install_cache("ss_cache", backend="sqlite", expire_after=900)

# -----------------------------------------------------------------------------
# Secrets & configuration helpers
# -----------------------------------------------------------------------------

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


OPENAI_API_KEY = (
    get_secret("OPENAI_API_KEY")           # вдруг ты так уже назвал
    or get_secret("OPENROUTER_API_KEY")    # ключ OpenRouter
)

OPENAI_MODEL = (
    get_secret("OPENAI_MODEL")             # если задашь руками
    or get_secret("OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free")
)

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
RAW_OLLAMA_URL = (get_secret("OLLAMA_URL") or "http://localhost:11434").strip()
OLLAMA_MODEL = get_secret("OLLAMA_MODEL", "mistral")



def resolve_ollama_endpoints(raw_url: str) -> Tuple[str, str]:
    base = (raw_url or "http://localhost:11434").strip()
    if not base:
        base = "http://localhost:11434"
    trimmed = base.rstrip("/")

    if trimmed.endswith("/api/version"):
        api_base = trimmed.rsplit("/", 1)[0]
        return f"{api_base}/generate", trimmed

    if trimmed.endswith("/api/generate"):
        api_base = trimmed.rsplit("/", 1)[0]
        return trimmed, f"{api_base}/version"

    if trimmed.endswith("/generate"):
        api_base = trimmed.rsplit("/", 1)[0]
        return trimmed, f"{api_base}/version"

    if trimmed.endswith("/api"):
        return f"{trimmed}/generate", f"{trimmed}/version"

    parsed = urlparse(trimmed)
    if parsed.path and parsed.path not in {"", "/"}:
        # Считаем, что путь указывает на endpoint generate
        parent = trimmed.rsplit("/", 1)[0]
        return trimmed, f"{parent}/version"

    api_base = f"{trimmed}/api"
    return f"{api_base}/generate", f"{api_base}/version"


OLLAMA_GENERATE_URL, OLLAMA_VERSION_URL = resolve_ollama_endpoints(RAW_OLLAMA_URL)
OLLAMA_CHAT_URL = OLLAMA_GENERATE_URL.rsplit("/", 1)[0] + "/chat"

WATCHLIST_FILE = Path(__file__).with_name("watchlist.json")
DEFAULT_WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "NFLX",
    "AMD",
    "IBM",
]

DEMO_EARNINGS_DATES = {
    "AAPL": "2024-08-02",
    "MSFT": "2024-07-25",
    "NVDA": "2024-08-15",
    "AMZN": "2024-07-30",
    "GOOGL": "2024-07-24",
    "META": "2024-07-31",
    "TSLA": "2024-07-23",
    "NFLX": "2024-07-18",
    "AMD": "2024-07-29",
    "IBM": "2024-07-22",
}

DEMO_NEWS: Dict[str, List[Dict[str, str]]] = {
    "AAPL": [
        {
            "headline": "Apple boosts services revenue on strong App Store demand",
            "summary": "Analysts highlight steady growth in subscriptions amid hardware softness.",
        },
        {
            "headline": "Suppliers hint at new wearable upgrades ahead of holiday season",
            "summary": "Component orders signal refreshed line-up with improved health sensors.",
        },
        {
            "headline": "Regulators review App Store terms in Europe",
            "summary": "Commission questions recent pricing changes and developer policies.",
        },
    ],
    "MSFT": [
        {
            "headline": "Microsoft expands AI partnerships with enterprise clients",
            "summary": "Azure OpenAI demand drives backlog of cloud migration projects.",
        },
        {
            "headline": "Xbox unit eyes subscription bundle refresh",
            "summary": "Gaming division tests new streaming-first offers to boost engagement.",
        },
        {
            "headline": "Competition authorities revisit Activision integration milestones",
            "summary": "UK and EU seek updates on compliance commitments.",
        },
    ],
    "NVDA": [
        {
            "headline": "Nvidia ramps H100 shipments as data centers upgrade",
            "summary": "Partners cite supply tightness persisting into next quarter.",
        },
        {
            "headline": "Automotive pipeline adds new autonomous driving wins",
            "summary": "Tier-1 suppliers choose Nvidia Drive for 2025 platforms.",
        },
        {
            "headline": "Export controls remain a headline risk in China",
            "summary": "Management reiterates mitigation plans with local variants.",
        },
    ],
    "AMZN": [
        {
            "headline": "Prime Day preview shows focus on home electronics",
            "summary": "Logistics teams report smooth execution tests across US regions.",
        },
        {
            "headline": "AWS unveils new cost-optimization tier for AI workloads",
            "summary": "Early adopters cite double-digit savings on inference jobs.",
        },
        {
            "headline": "FTC litigation schedule slips into year-end",
            "summary": "Court sets fall timeline for hearings on marketplace practices.",
        },
    ],
    "GOOGL": [
        {
            "headline": "Google integrates Gemini into Workspace pricing tiers",
            "summary": "Enterprise clients see bundling with security add-ons.",
        },
        {
            "headline": "YouTube Shorts monetization trending higher",
            "summary": "Creators note improved revenue share amid advertiser experiments.",
        },
        {
            "headline": "DoJ antitrust closing arguments scheduled",
            "summary": "Search market share remains central to regulatory debate.",
        },
    ],
    "META": [
        {
            "headline": "Meta tests AI-generated ads tools across SMB accounts",
            "summary": "Marketers highlight quicker creative iteration.",
        },
        {
            "headline": "Reality Labs spending path moderated",
            "summary": "Company signals disciplined rollout of mixed reality devices.",
        },
        {
            "headline": "Privacy updates in EU messaging products",
            "summary": "Regulators monitor interoperability commitments closely.",
        },
    ],
    "TSLA": [
        {
            "headline": "Tesla launches updated Model Y incentives in Europe",
            "summary": "Pricing actions aim to defend share amid new entrants.",
        },
        {
            "headline": "Cybertruck production ramp tracking internal goals",
            "summary": "Suppliers cite stable battery output in Austin.",
        },
        {
            "headline": "Autopilot investigations stay in focus",
            "summary": "Regulators request additional driver-assistance data sets.",
        },
    ],
    "NFLX": [
        {
            "headline": "Netflix scales ad-tier inventory in Latin America",
            "summary": "Launch partners note strong premium brand interest.",
        },
        {
            "headline": "Content slate leans into live events for fall",
            "summary": "Sports docuseries and award shows targeted for engagement.",
        },
        {
            "headline": "Password sharing crackdown metrics positive",
            "summary": "New sign-ups exceed churn in latest weekly data.",
        },
    ],
    "AMD": [
        {
            "headline": "AMD previews next-gen MI325 accelerators",
            "summary": "Roadmap keeps cadence with hyperscale demand curve.",
        },
        {
            "headline": "PC channel inventories normalizing",
            "summary": "Partners expect seasonal rebound into back-to-school period.",
        },
        {
            "headline": "Semi-custom pipeline expands beyond gaming",
            "summary": "Edge AI and telecom customers evaluate embedded solutions.",
        },
    ],
    "IBM": [
        {
            "headline": "IBM extends hybrid cloud partnerships with banks",
            "summary": "Regulated customers adopt secure fintech tooling.",
        },
        {
            "headline": "Quantum roadmap update highlights error mitigation",
            "summary": "Research unit showcases new milestone for stability.",
        },
        {
            "headline": "Consulting backlog sees public sector wins",
            "summary": "Government digital transformation projects remain active.",
        },
    ],
}

DEMO_NEWS_DEFAULT = [
    {
        "headline": "Company maintains stable outlook amid mixed macro signals",
        "summary": "Management balances investment with disciplined cost control.",
    },
    {
        "headline": "Industry peers note resilient demand trends",
        "summary": "Channel checks highlight steady pricing and healthy backlog.",
    },
]

DEMO_RISK_BULLETS = [
    "Регуляторы: Возможное ужесточение требований и штрафы",
    "Конкуренция: Давление со стороны сильных игроков и новых продуктов",
    "Исполнение: Риск срыва дорожной карты и контроля расходов",
]

SOURCE_LABELS = {
    "Finnhub": "Finnhub (основной API)",
    "Stooq": "Stooq (резервный канал)",
    "Demo": "Демо-данные (офлайн-режим)",
}

WATCHLIST_FILTER_LABELS = {
    "Все тикеры": "All",
    "Скоро отчётность": "Earnings soon",
    "Высокий новостной фон": "High news flow",
}

WATCHLIST_FILTER_OPTIONS = list(WATCHLIST_FILTER_LABELS.keys())

PROVIDER_OPTION_MAP = {
    "Авто": "Auto",
    "OpenRouter": "OpenAI",   # внутри всё равно 'openai'
    "Ollama": "Ollama",
    "Baseline": "Baseline",
}


SENTIMENT_CLASS_MAP = {
    "Позитив": "positive",
    "Негатив": "negative",
    "Нейтральная": "neutral",
}

CUSTOM_CSS = """
<style>
main .block-container {
    padding-top: 1.5rem;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 18px;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.1);
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.kpi-card .kpi-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
}

.kpi-card .kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
}

.kpi-card .kpi-extra {
    font-size: 0.9rem;
    color: #334155;
}

.kpi-card.positive .kpi-value {
    color: #15803d;
}

.kpi-card.negative .kpi-value {
    color: #b91c1c;
}

.ss-section {
    background: #ffffff;
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.12);
    border: 1px solid #e2e8f0;
    margin-bottom: 24px;
}

.ss-section h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-weight: 700;
    color: #0f172a;
}

.ss-section h4 {
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    color: #1e293b;
}

.ss-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e2e8f0;
    color: #334155;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-bottom: 12px;
}

.ss-highlight-card {
    background: #f8fafc;
    border-radius: 14px;
    padding: 16px 18px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
}

.ss-highlight-card h4 {
    margin: 0 0 6px 0;
    font-size: 1.05rem;
    color: #0f172a;
}

.ss-highlight-card p {
    margin: 0 0 10px 0;
    color: #475569;
}

.ss-list {
    margin: 0;
    padding-left: 1.1rem;
    color: #334155;
}

.ss-sentiment-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    margin-bottom: 12px;
}

.ss-sentiment-pill.positive {
    background: rgba(34, 197, 94, 0.18);
    color: #15803d;
}

.ss-sentiment-pill.negative {
    background: rgba(248, 113, 113, 0.2);
    color: #b91c1c;
}

.ss-sentiment-pill.neutral {
    background: rgba(148, 163, 184, 0.2);
    color: #334155;
}

.ss-forecast {
    background: #f8fafc;
    border-radius: 12px;
    padding: 12px 14px;
    border: 1px dashed #cbd5f5;
    color: #1e293b;
    font-size: 0.95rem;
    word-break: break-word;
}

.ss-risks li {
    margin-bottom: 6px;
}

.sidebar-badge {
    display: block;
    padding: 6px 10px;
    border-radius: 10px;
    background: #e2e8f0;
    margin-bottom: 6px;
    font-size: 0.9rem;
}

.sidebar-badge strong {
    color: #0f172a;
}
</style>
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_today() -> date:
    return utc_now().date()


def translate_source(name: str) -> str:
    return SOURCE_LABELS.get(name, name)


def sentiment_css_class(sentiment: str) -> str:
    return SENTIMENT_CLASS_MAP.get(sentiment, "neutral")


def unique_ordered(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for raw in items:
        if not raw:
            continue
        cleaned = re.sub(r"\s+", " ", str(raw).strip())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def build_system_prompt(ticker: str, context: Dict[str, object], use_context: bool) -> str:
    quote = context.get("quote", {}) if context else {}
    analysis = context.get("analysis", {}) if context else {}
    price = quote.get("price")
    pct = quote.get("pct")
    sent = analysis.get("sentiment")
    kps = ", ".join(analysis.get("keypoints", [])[:3]) or "—"
    risks = ", ".join(analysis.get("risks", [])[:3]) or "—"

    ctx = f"""
    <CONTEXT>
    Ticker: {ticker}
    Price: {price if price is not None else '—'}; Δ%: {pct if pct is not None else '—'}
    Sentiment: {sent or 'Нейтральная'}
    Keypoints: {kps}
    Risks: {risks}
    </CONTEXT>
    """.strip()

    return (
        "Ты — StockSense, инвестиционный ИИ-помощник. "
        "Отвечай лаконично, всегда на русском структурно и по-делу для финансового аналитика. "
        "Всегда указывай допущения и горизонты, избегай категоричных прогнозов. "
        "Формат по умолчанию: TL;DR (1–2 фразы) • Драйверы • Риски • Следующие шаги (конкретные действия на 1–2 недели). "
        "Если данных мало — явно это скажи и предложи, что ещё уточнить. "
        + (ctx if use_context else "")
    )
# -----------------------------------------------------------------------------
# LLM provider selection and helpers
# -----------------------------------------------------------------------------

@st.cache_data(ttl=120)
def is_ollama_available() -> bool:
    try:
        resp = requests.get(OLLAMA_VERSION_URL, timeout=2)
        return resp.ok
    except Exception:
        return False


@st.cache_data(ttl=120)
def is_ollama_chat_available() -> bool:
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": "ping"}],
                "stream": False,
            },
            timeout=3,
        )
        return resp.ok
    except Exception:
        return False


def has_openai() -> bool:
    return bool(OPENAI_API_KEY)


def provider_priority() -> List[str]:
    return ["openai", "ollama", "baseline"]


def determine_active_provider(use_llm: bool, forced: str) -> Tuple[str, str]:
    if not use_llm:
        return "baseline", "LLM отключён — используем детерминированные эвристики."

    forced = (forced or "auto").lower()
    order: List[str] = []
    if forced == "auto":
        order = provider_priority()
    else:
        order = [forced] + [p for p in provider_priority() if p != forced]

    for provider in order:
        if provider == "openai" and has_openai():
            return "openai", "Используется OpenAI Chat Completions."
        if provider == "ollama" and is_ollama_available():
            return "ollama", "Доступен локальный Ollama — переключаемся на него."
        if provider == "baseline":
            return "baseline", "Работаем на встроенных эвристиках."

    return "baseline", "Внешние LLM-провайдеры недоступны."


def ordered_providers(preferred: Optional[str]) -> Iterable[str]:
    preferred = (preferred or "auto").lower()
    if preferred in {"auto", ""}:
        return provider_priority()
    return [preferred] + [p for p in provider_priority() if p != preferred]


def _openai_request(messages: List[Dict[str, str]], max_tokens: int = 300) -> Optional[str]:
    """
    На самом деле ходит в OpenRouter, но слот 'openai' оставляем,
    чтобы не переписывать остальной код.
    """
    if not has_openai():
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv(
            "OPENROUTER_SITE_URL",
            "https://dimixl-stocktest.streamlit.app",  # <-- поменяй на URL своего аппа
        ),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "StockSense"),
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        # временный вывод для дебага
        if not resp.ok:
            # не логируем ключи, только статус и текст
            st.error(f"LLM error: {resp.status_code} {resp.text[:500]}")
            return None

        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
    except Exception as e:
        st.error(f"LLM exception: {e}")
        return None




def _ollama_request(prompt: str) -> Optional[str]:
    if not is_ollama_available():
        return None
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=20)
        if resp.ok:
            data = resp.json()
            return data.get("response")
    except Exception:
        return None
    return None


def _format_chat_messages(messages: List[Dict[str, str]]) -> str:
    formatted: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n\n".join(formatted)


def _ollama_chat(messages: List[Dict[str, str]], timeout: int = 30) -> Optional[str]:
    if not is_ollama_available():
        return None
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    try:
        if is_ollama_chat_available():
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
            if resp.ok:
                data = resp.json()
                message_block = data.get("message")
                if isinstance(message_block, dict):
                    content = message_block.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        nested = first.get("message")
                        if isinstance(nested, dict):
                            content = nested.get("content")
                            if isinstance(content, str) and content.strip():
                                return content
        prompt = _format_chat_messages(messages)
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.ok:
            data = resp.json()
            content = data.get("response")
            if isinstance(content, str) and content.strip():
                return content
    except Exception:
        return None
    return None


def trim_chat_history(history: List[Dict[str, str]], limit: int = 20) -> None:
    if len(history) > limit:
        del history[:-limit]


POSITIVE_TERMS = {
    "growth",
    "beat",
    "strong",
    "up",
    "positive",
    "bullish",
    "record",
    "outperform",
    "upgrade",
    "buyback",
    "improve",
}
NEGATIVE_TERMS = {
    "weak",
    "down",
    "negative",
    "bearish",
    "lawsuit",
    "downgrade",
    "risk",
    "guidance cut",
    "investigation",
    "decline",
    "loss",
}
KEYPOINT_KEYWORDS = [
    "revenue",
    "guidance",
    "profit",
    "margin",
    "lawsuit",
    "upgrade",
    "downgrade",
    "buyback",
    "demand",
    "supply",
]


def re_split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def sentiment_lexicon(text: str) -> str:
    lowered = text.lower()
    pos = sum(lowered.count(term) for term in POSITIVE_TERMS)
    neg = sum(lowered.count(term) for term in NEGATIVE_TERMS)
    if pos > neg:
        return "Позитив"
    if neg > pos:
        return "Негатив"
    return "Нейтральная"


def keypoints_regex(text: str, n: int = 3) -> List[str]:
    sentences = [s.strip() for s in re_split_sentences(text) if s.strip()]
    scored: List[Tuple[int, str]] = []
    for sent in sentences:
        score = sum(1 for kw in KEYPOINT_KEYWORDS if kw in sent.lower())
        scored.append((score, sent))
    scored.sort(key=lambda item: (-item[0], sentences.index(item[1])))
    top = [s for _, s in scored if s][:n]
    if len(top) < n:
        top.extend(sentences[: n - len(top)])
    return top[:n]


def llm_chat_completion(
    messages: List[Dict[str, str]], provider: Optional[str] = None, max_tokens: int = 400
) -> Optional[str]:
    for prov in ordered_providers(provider):
        if prov == "openai":
            response = _openai_request(messages, max_tokens=max_tokens)
            if response:
                return response.strip()
        elif prov == "ollama":
            response = _ollama_chat(messages)
            if response:
                return response.strip()
    return None


def llm_sentiment(text: str, provider: Optional[str] = None) -> str:
    for prov in ordered_providers(provider):
        if prov == "openai":
            response = _openai_request(
                [
                    {
                        "role": "system",
                        "content": "Определи тональность текста. Ответь одним словом: Позитив, Нейтральная или Негатив.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=10,
            )
            if response:
                normalized = response.strip().split()[0]
                if "поз" in normalized.lower():
                    return "Позитив"
                if "нег" in normalized.lower():
                    return "Негатив"
                return "Нейтральная"
        elif prov == "ollama":
            prompt = (
                "Тональность текста? Ответь одним словом из списка: Позитив, Нейтральная, Негатив.\n"
                + text
            )
            response = _ollama_request(prompt)
            if response:
                normalized = response.strip().split()[0]
                if "поз" in normalized.lower():
                    return "Позитив"
                if "нег" in normalized.lower():
                    return "Негатив"
                return "Нейтральная"
        else:
            return sentiment_lexicon(text)
    return sentiment_lexicon(text)


def llm_keypoints(text: str, n: int = 3, provider: Optional[str] = None) -> List[str]:
    for prov in ordered_providers(provider):
        if prov == "openai":
            response = _openai_request(
                [
                    {"role": "system", "content": "Выдели ключевые тезисы. Дай краткие буллеты."},
                    {"role": "user", "content": f"Тезисы (максимум {n}): {text}"},
                ],
                max_tokens=120,
            )
            if response:
                bullets = [line.strip("-• ") for line in response.strip().splitlines() if line.strip()]
                return bullets[:n] if bullets else [response.strip()][:n]
        elif prov == "ollama":
            prompt = (
                f"Выдели максимум {n} ключевых тезисов из текста. Формат: по одному тезису в строке.\n{text}"
            )
            response = _ollama_request(prompt)
            if response:
                bullets = [line.strip("-• ") for line in response.strip().splitlines() if line.strip()]
                return bullets[:n] if bullets else [response.strip()][:n]
        else:
            return keypoints_regex(text, n=n)
    return keypoints_regex(text, n=n)


def llm_chat(prompt: str, provider: Optional[str] = None) -> str:
    for prov in ordered_providers(provider):
        if prov == "openai":
            response = _openai_request(
                [
                    {"role": "system", "content": "Дай краткий ответ для финансового аналитика."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            if response:
                return response.strip()
        elif prov == "ollama":
            response = _ollama_request(
                prompt + "\nОтветь лаконично, 2-3 предложения, для финансового аналитика."
            )
            if response:
                return response.strip()
        else:
            return (
                "Base: Стабильный тренд ожидается | Upside: Возможен рост при сильных новостях | "
                "Downside: Следить за макро-волатильностью (1–2 недели)"
            )
    return (
        "Base: Стабильный тренд ожидается | Upside: Возможен рост при сильных новостях | "
        "Downside: Следить за макро-волатильностью (1–2 недели)"
    )
# -----------------------------------------------------------------------------
# Market data utilities
# -----------------------------------------------------------------------------


def generate_demo_series(symbol: str, count: int = 300) -> pd.DataFrame:
    seed = abs(hash(symbol)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    base_price = 50 + (seed % 150)
    returns = rng.normal(loc=0.0005, scale=0.02, size=count)
    prices = base_price * np.exp(np.cumsum(returns))
    times = pd.date_range(end=utc_now(), periods=count, freq="B")
    opens = np.maximum(prices * (1 + rng.normal(0, 0.01, size=count)), 1)
    highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0.002, 0.01, size=count)))
    lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0.002, 0.01, size=count)))
    volumes = rng.integers(1_000_000, 5_000_000, size=count)
    df = pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )
    df = df.sort_values("time")
    return df


@st.cache_data(ttl=600)
def fetch_history(symbol: str, count: int = 300) -> Tuple[pd.DataFrame, str]:
    # 1) Пытаемся через Finnhub, как и раньше
    if FINNHUB_API_KEY:
        params = {
            "symbol": symbol,
            "resolution": "D",
            "count": count,
            "token": FINNHUB_API_KEY,
        }
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/candle",
                params=params,
                timeout=5,
            )
            data = resp.json()
            if data.get("s") == "ok" and data.get("t"):
                df = pd.DataFrame(
                    {
                        "time": pd.to_datetime(data["t"], unit="s"),
                        "open": data["o"],
                        "high": data["h"],
                        "low": data["l"],
                        "close": data["c"],
                        "volume": data.get("v", [0] * len(data["t"])),
                    }
                )
                df = df.sort_values("time")
                return df, "Finnhub"
        except Exception:
            pass

    # 2) Фолбэк на Stooq через CSV (без pandas_datareader)
    try:
        # Stooq ожидает тикер вида aapl.us
        sym = symbol.lower()
        if "." not in sym:
            sym = sym + ".us"

        url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.rename(
                columns={
                    "Date": "time",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            ).sort_values("time")

            # Обрежем до нужного количества свечей
            if len(df) > count:
                df = df.tail(count)

            return df[["time", "open", "high", "low", "close", "volume"]], "Stooq"
    except Exception:
        pass

    # 3) Если всё отвалилось — демо-серия
    return generate_demo_series(symbol, count=count), "Demo"



def fetch_quote(symbol: str) -> Tuple[Dict[str, Optional[float]], str]:
    quote: Dict[str, Optional[float]] = {"price": None, "pct": None, "timestamp": None}
    if FINNHUB_API_KEY:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": FINNHUB_API_KEY},
                timeout=5,
            )
            data = resp.json()
            if data:
                price = data.get("c")
                prev = data.get("pc")
                if price is not None:
                    quote["price"] = float(price)
                if prev not in (None, 0):
                    quote["pct"] = round(((price or 0) - prev) / prev * 100, 2)
                ts = data.get("t")
                if ts:
                    quote["timestamp"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                if quote["price"] is not None:
                    return quote, "Finnhub"
        except Exception:
            pass

    history, source = fetch_history(symbol, count=5)
    if not history.empty:
        price = history["close"].iloc[-1]
        prev = history["close"].iloc[-2] if len(history) > 1 else price
        quote["price"] = round(float(price), 2)
        quote["pct"] = round((price - prev) / prev * 100, 2) if prev else None
        quote["timestamp"] = history["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        return quote, source

    demo = generate_demo_series(symbol, count=10)
    price = demo["close"].iloc[-1]
    prev = demo["close"].iloc[-2]
    quote["price"] = round(float(price), 2)
    quote["pct"] = round((price - prev) / prev * 100, 2)
    quote["timestamp"] = demo["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    return quote, "Demo"


@st.cache_data(ttl=600)
def pull_company_news(symbol: str) -> Tuple[List[Dict[str, str]], str]:
    items: List[Dict[str, str]] = []
    if FINNHUB_API_KEY:
        try:
            today = utc_today()
            start = today - timedelta(days=7)
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from": start.isoformat(),
                    "to": today.isoformat(),
                    "token": FINNHUB_API_KEY,
                },
                timeout=5,
            )
            data = resp.json()
            for item in data[:5]:
                if item.get("headline"):
                    items.append(
                        {
                            "headline": item.get("headline", ""),
                            "summary": item.get("summary") or item.get("headline", ""),
                            "datetime": item.get("datetime"),
                            "url": item.get("url", ""),
                        }
                    )
            if items:
                return items, "Finnhub"
        except Exception:
            pass

    demo_items = DEMO_NEWS.get(symbol.upper(), DEMO_NEWS_DEFAULT)
    now = utc_now()
    enriched = []
    for idx, item in enumerate(demo_items[:5]):
        enriched.append(
            {
                "headline": item["headline"],
                "summary": item["summary"],
                "datetime": (now - timedelta(hours=6 * idx)).strftime("%Y-%m-%d %H:%M:%S"),
                "url": "",
            }
        )
    return enriched, "Demo"


@st.cache_data(ttl=600)
def compute_volatility(symbol: str) -> float:
    history, _ = fetch_history(symbol, count=60)
    if history.empty:
        return 0.0
    closes = history["close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna().tail(5)
    if returns.empty:
        return 0.0
    return float(returns.std())


# -----------------------------------------------------------------------------
# Analytics helpers
# -----------------------------------------------------------------------------


def analyze_news_items(
    symbol: str,
    news_items: List[Dict[str, str]],
    provider: str,
) -> Dict[str, object]:
    highlights = []
    sentiments: List[str] = []
    aggregate_keypoints: List[str] = []
    for item in news_items[:5]:
        text = f"{item['headline']}. {item.get('summary', '')}"
        sentiment = llm_sentiment(text, provider=provider)
        keypoints = unique_ordered(llm_keypoints(text, n=3, provider=provider))
        sentiments.append(sentiment)
        aggregate_keypoints.extend(keypoints)
        highlights.append(
            {
                "headline": item["headline"],
                "summary": item.get("summary", ""),
                "datetime": item.get("datetime"),
                "url": item.get("url", ""),
                "sentiment": sentiment,
                "keypoints": keypoints[:3],
            }
        )

    sentiment_summary = "Нейтральная"
    if sentiments:
        pos = sentiments.count("Позитив")
        neg = sentiments.count("Негатив")
        if pos > neg:
            sentiment_summary = "Позитив"
        elif neg > pos:
            sentiment_summary = "Негатив"

    aggregate_keypoints = unique_ordered(aggregate_keypoints)
    keypoints = (
        aggregate_keypoints[:3]
        if aggregate_keypoints
        else [item["headline"] for item in news_items[:3]]
    )
    keypoints = unique_ordered(keypoints)[:3]
    if len(keypoints) < 3:
        keypoints.extend(["Тренд спроса", "Фокус на маржинальности", "Контроль расходов"][len(keypoints) : 3])

    if provider == "baseline":
        risks = DEMO_RISK_BULLETS[:3]
    else:
        risk_prompt = (
            f"Ticker: {symbol}. На основе последних новостей сформулируй 3 ключевых риска."
            " Каждый пункт — короткая фраза."
        )
        risks = unique_ordered(llm_keypoints(risk_prompt, n=3, provider=provider))
        if not risks:
            risks = DEMO_RISK_BULLETS[:3]

    forecast_prompt = (
        f"Сформируй мини-прогноз для {symbol}: формат Base | Upside | Downside (1–2 недели)."
        " Будь кратким."
    )
    forecast = llm_chat(forecast_prompt, provider=provider)
    if "Base:" not in forecast:
        forecast = (
            "Base: Стабильный тренд | Upside: Рост при сильной выручке | "
            "Downside: Снижение при слабых новостях (1–2 недели)"
        )

    return {
        "highlights": highlights,
        "sentiment": sentiment_summary,
        "keypoints": keypoints[:3],
        "risks": risks[:3],
        "forecast": forecast,
    }


def gather_ticker_context(symbol: str, provider: str) -> Dict[str, object]:
    quote, quote_source = fetch_quote(symbol)
    history, history_source = fetch_history(symbol)
    news_items, news_source = pull_company_news(symbol)
    analysis = analyze_news_items(symbol, news_items, provider)
    return {
        "ticker": symbol,
        "quote": quote,
        "quote_source": quote_source,
        "history": history,
        "history_source": history_source,
        "news_items": news_items,
        "news_source": news_source,
        "analysis": analysis,
        "provider": provider,
    }


def build_api_payload(context: Dict[str, object]) -> Dict[str, object]:
    quote = context["quote"]
    analysis = context["analysis"]
    keypoints = list(analysis.get("keypoints", []))[:3]
    risks = list(analysis.get("risks", []))[:3]
    while len(keypoints) < 3:
        keypoints.append("—")
    while len(risks) < 3:
        risks.append("—")
    last_update = quote.get("timestamp")
    if not last_update:
        history: pd.DataFrame = context["history"]
        if not history.empty:
            last_update = history["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": context["ticker"],
        "price": quote.get("price"),
        "pct": quote.get("pct"),
        "last_update": last_update,
        "sentiment": analysis.get("sentiment"),
        "keypoints": keypoints,
        "risks": risks,
        "forecast_text": analysis.get("forecast"),
    }


def render_candles(df: pd.DataFrame) -> go.Figure:
    price_trace = go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Цена",
    )
    volume_trace = go.Bar(
        x=df["time"],
        y=df["volume"],
        name="Объём",
        marker_color="#1f77b4",
        opacity=0.3,
        yaxis="y2",
    )
    fig = go.Figure(data=[price_trace, volume_trace])
    fig.update_layout(
        xaxis=dict(type="date", rangeslider=dict(visible=False)),
        yaxis=dict(title="Цена"),
        yaxis2=dict(title="Объём", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        margin=dict(l=30, r=30, t=40, b=40),
    )
    return fig


def run_backtest(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    if df.empty or fast >= slow:
        return pd.DataFrame()
    closes = df.set_index("time")["close"].astype(float)
    fast_ma = closes.rolling(window=fast).mean()
    slow_ma = closes.rolling(window=slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    returns = closes.pct_change().fillna(0)
    strategy_returns = returns * signal.shift(1).fillna(0)
    equity = (1 + strategy_returns).cumprod()
    result = pd.DataFrame({"time": closes.index, "equity": equity})
    return result.dropna()


def export_png(fig: go.Figure, header: Dict[str, str]) -> bytes:
    try:
        import plotly.io as pio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotly not available for export") from exc

    fig_copy = go.Figure(fig)
    header_text = " | ".join(f"{k}: {v}" for k, v in header.items())
    fig_copy.add_annotation(
        text=f"StockSense • Карточка тикера — {header_text}",
        x=0.5,
        xref="paper",
        y=1.12,
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#222"),
    )
    fig_copy.update_layout(margin=dict(t=120))
    buffer = io.BytesIO()
    try:
        pio.write_image(fig_copy, buffer, format="png", width=1100, height=700)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Kaleido is required for PNG export") from exc
    buffer.seek(0)
    return buffer.read()


def export_html_card(context: Dict[str, object]) -> str:
    quote = context["quote"]
    analysis = context["analysis"]
    highlights_html = "".join(
        f"<li><strong>{item['headline']}</strong><br/><em>{item['sentiment']}</em> — {item['summary']}</li>"
        for item in analysis.get("highlights", [])
    ) or "<li>—</li>"
    risks_html = "".join(f"<li>{risk}</li>" for risk in analysis.get("risks", [])) or "<li>—</li>"
    keypoints_html = "".join(f"<li>{kp}</li>" for kp in analysis.get("keypoints", [])) or "<li>—</li>"
    price = format_price(quote.get("price"))
    pct = quote.get("pct")
    pct_display = f"{pct:+.2f}%" if isinstance(pct, (int, float, np.floating)) else "—"
    generated_at = utc_now().strftime("%Y-%m-%d %H:%M UTC")
    template = f"""
    <html>
    <head>
        <meta charset='utf-8'/>
        <title>StockSense • Карточка {context['ticker']}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #1f2933; background: #f8fafc; }}
            h1 {{ color: #0f172a; margin-bottom: 12px; }}
            section {{ margin-bottom: 20px; background: #ffffff; padding: 18px 22px; border-radius: 16px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08); }}
            .kpi {{ display: flex; gap: 24px; flex-wrap: wrap; }}
            .kpi div {{ background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%); padding: 14px 18px; border-radius: 12px; min-width: 180px; box-shadow: inset 0 0 0 1px rgba(14, 116, 144, 0.1); }}
            .kpi div strong {{ display: block; margin-bottom: 6px; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.04em; color: #0f172a; }}
            ul {{ padding-left: 20px; margin: 0; }}
            footer {{ margin-top: 28px; font-size: 12px; color: #64748b; text-align: right; }}
        </style>
    </head>
    <body>
        <h1>StockSense • Карточка тикера</h1>
        <section class='kpi'>
            <div><strong>Тикер</strong>{context['ticker']}</div>
            <div><strong>Цена</strong>{price}</div>
            <div><strong>Δ%</strong>{pct_display}</div>
            <div><strong>Тональность</strong>{analysis.get('sentiment', 'Нейтральная')}</div>
        </section>
        <section>
            <h2>Главные тезисы</h2>
            <ul>{keypoints_html}</ul>
        </section>
        <section>
            <h2>Новости</h2>
            <ul>{highlights_html}</ul>
        </section>
        <section>
            <h2>Ключевые риски</h2>
            <ul>{risks_html}</ul>
        </section>
        <section>
            <h2>Мини-прогноз</h2>
            <p>{analysis.get('forecast', 'Нет данных')}</p>
        </section>
        <footer>Сформировано StockSense {generated_at}</footer>
    </body>
    </html>
    """
    return template


def api_payload(symbol: str, provider: str) -> Dict[str, object]:
    context = gather_ticker_context(symbol, provider)
    return build_api_payload(context)
# -----------------------------------------------------------------------------
# Watchlist persistence helpers
# -----------------------------------------------------------------------------


def load_watchlist() -> List[str]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text())
            if isinstance(data, list):
                symbols = [str(item).upper() for item in data if str(item).strip()]
                return symbols or DEFAULT_WATCHLIST.copy()
        except Exception:
            pass
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(symbols: List[str]) -> None:
    try:
        WATCHLIST_FILE.write_text(json.dumps(symbols, indent=2))
    except Exception:
        pass


def get_watchlist_state() -> List[str]:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()
    return st.session_state.watchlist


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def format_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def render_kpi_cards(
    quote: Dict[str, Optional[float]], history: pd.DataFrame, context: Dict[str, object]
) -> None:
    price_value = format_price(quote.get("price"))
    pct = quote.get("pct")
    pct_value = "—"
    card_class = "kpi-card"
    if isinstance(pct, (int, float, np.floating)):
        pct_value = f"{pct:+.2f}%"
        if pct > 0:
            card_class += " positive"
        elif pct < 0:
            card_class += " negative"
    last_update = quote.get("timestamp")
    if not last_update and not history.empty:
        last_update = history["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    history_count = len(history.index) if not history.empty else 0
    last_bar = (
        history["time"].iloc[-1].strftime("%Y-%m-%d") if history_count else "—"
    )
    quote_source = translate_source(str(context.get("quote_source", "—")))
    history_source = translate_source(str(context.get("history_source", "—")))
    cards_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="kpi-label">Текущая цена</span>
            <span class="kpi-value">{price_value}</span>
            <span class="kpi-extra">по тикеру {context['ticker']}</span>
        </div>
        <div class="{card_class}">
            <span class="kpi-label">Изменение</span>
            <span class="kpi-value">{pct_value}</span>
            <span class="kpi-extra">к предыдущему закрытию</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Обновлено</span>
            <span class="kpi-value">{last_update or '—'}</span>
            <span class="kpi-extra">Источник цены: {quote_source}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">История</span>
            <span class="kpi-value">{history_count or '—'} свечей</span>
            <span class="kpi-extra">{history_source} • последнее: {last_bar}</span>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)



def render_highlights_section(container, analysis: Dict[str, object], news_source: str) -> None:
    highlight_blocks: List[str] = []
    for item in analysis.get("highlights", []):
        keypoints = unique_ordered(item.get("keypoints", []))
        keypoints_html = "".join(f"<li>{kp}</li>" for kp in keypoints)
        list_html = f"<ul class='ss-list'>{keypoints_html}</ul>" if keypoints_html else ""
        sentiment = item.get("sentiment", "Нейтральная")
        sentiment_class = sentiment_css_class(sentiment)

        block = dedent(f"""
        <div class="ss-highlight-card">
        <span class="ss-sentiment-pill {sentiment_class}">{sentiment}</span>
        <h4>{item.get('headline', 'Новость')}</h4>
        <p>{item.get('summary', '')}</p>
        {list_html}
        </div>
        """).strip()

        highlight_blocks.append(block)

    if not highlight_blocks:
        highlight_blocks.append("<p>Свежие новости не найдены.</p>")

    keypoints_html = "".join(f"<li>{kp}</li>" for kp in analysis.get("keypoints", []) if kp) or "<li>—</li>"

    html = dedent(f"""
    <div class="ss-section">
    <h3>Ключевые события</h3>
    <span class="ss-badge"><strong>Источник:</strong> {translate_source(news_source)}</span>
    {''.join(highlight_blocks)}
    <h4>Главные тезисы</h4>
    <ul class="ss-list">{keypoints_html}</ul>
    </div>
    """).strip()

    container.markdown(html, unsafe_allow_html=True)


def render_sentiment_section(container, analysis: Dict[str, object]) -> None:
    sentiment = analysis.get("sentiment") or "Нейтральная"
    sentiment_class = sentiment_css_class(sentiment)
    risks_html = "".join(f"<li>{risk}</li>" for risk in analysis.get("risks", []) if risk)
    if not risks_html:
        risks_html = "<li>—</li>"
    forecast = analysis.get("forecast") or "Нет данных"
    container.markdown(
        f"""
        <div class="ss-section">
            <h3>Общая тональность</h3>
            <div class="ss-sentiment-pill {sentiment_class}">{sentiment}</div>
            <h4>Основные риски</h4>
            <ul class="ss-list ss-risks">{risks_html}</ul>
            <h4>Мини-прогноз</h4>
            <div class="ss-forecast">{forecast}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def filter_watchlist(symbols: List[str], preset: str) -> List[str]:
    canonical = WATCHLIST_FILTER_LABELS.get(preset, preset or "All")
    if canonical == "All":
        return symbols
    if canonical == "Earnings soon":
        today = utc_today()
        filtered = []
        for symbol in symbols:
            date_str = DEMO_EARNINGS_DATES.get(symbol.upper())
            if not date_str:
                continue
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if abs((event_date - today).days) <= 14:
                filtered.append(symbol)
        return filtered or symbols
    if canonical == "High news flow":
        vol_pairs = [(symbol, compute_volatility(symbol)) for symbol in symbols]
        vol_pairs.sort(key=lambda item: item[1], reverse=True)
        top = [symbol for symbol, vol in vol_pairs if vol > 0][: max(5, len(symbols) // 2 or 1)]
        return top or symbols
    return symbols


def render_watchlist_table(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        quote, _ = fetch_quote(symbol)
        rows.append(
            {
                "Тикер": symbol,
                "Цена": format_price(quote.get("price")),
                "Δ%": f"{quote.get('pct'):.2f}%" if quote.get("pct") is not None else "—",
                "Обновлено": quote.get("timestamp") or "—",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        return df.set_index("Тикер")
    return df


def reset_backtest_state(symbol: str) -> None:
    backtests = st.session_state.setdefault("backtests", {})
    params = st.session_state.setdefault("backtest_params", {})
    backtests.pop(symbol, None)
    params.pop(symbol, None)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main() -> None:
    params = st.query_params
    ticker_param = params.get("ticker")
    requested_symbol = (ticker_param[0] if ticker_param else DEFAULT_WATCHLIST[0]).upper()
    api_param = params.get("api")
    use_api_mode = bool(api_param and api_param[0] == "1")

    watchlist = get_watchlist_state()
    if not watchlist:
        watchlist = DEFAULT_WATCHLIST.copy()
        st.session_state.watchlist = watchlist

    if use_api_mode:
        active_provider, _ = determine_active_provider(True, "Auto")
        symbol = requested_symbol or (watchlist[0] if watchlist else "AAPL")
        payload = api_payload(symbol, active_provider)
        st.json(payload)
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## StockSense")
        st.caption("Мгновенная аналитика по акциям")
        use_llm = st.toggle("Использовать LLM", value=True)
        provider_label = st.selectbox("Провайдер LLM", list(PROVIDER_OPTION_MAP.keys()), index=0)
        provider_choice = PROVIDER_OPTION_MAP.get(provider_label, "Auto")
        active_provider, provider_message = determine_active_provider(use_llm, provider_choice)
        color_map = {"openai": "#16a34a", "ollama": "#d97706", "baseline": "#64748b"}
        provider_name_map = {"openai": "OpenAI", "ollama": "Ollama", "baseline": "Baseline"}
        provider_display = provider_name_map.get(active_provider, active_provider.title())
        color_value = color_map.get(active_provider, "#64748b")
        st.markdown(
            f"**LLM-провайдер:** <span style='color:{color_value}; font-weight:600'>{provider_display}</span>",
            unsafe_allow_html=True,
        )
        st.caption(provider_message)
        if use_llm and active_provider != "openai":
            if active_provider == "ollama":
                st.info("OpenAI недоступен — используем локальный Ollama (режим пониженной готовности).")
            else:
                st.warning("Работаем на baseline-эвристиках без внешнего LLM.")
        st.markdown("### 🔑 Доступы")
        st.markdown(
            f"<div class='sidebar-badge'><strong>OpenRouter:</strong> {'✅ Активен' if has_openai() else '⚪️ Нет — используем резерв'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-badge'><strong>Finnhub:</strong> {'✅ Активен' if FINNHUB_API_KEY else '⚪️ Нет — переключимся на Stooq/демо'}</div>",
            unsafe_allow_html=True,
        )
        st.caption("⚪️ Нет — значит ключ не найден, задействуем резервные источники данных.")
        st.markdown("### 📊 Источники данных")
        quote_source_placeholder = st.empty()
        history_source_placeholder = st.empty()
        news_source_placeholder = st.empty()
        st.caption(f"DEBUG: has_openai={has_openai()} model={OPENAI_MODEL}")

    st.title("StockSense • Аналитика по акциям")
    st.caption("Карточка тикера и управляемый список наблюдения")
    tab_ticker, tab_watch, tab_chat = st.tabs(["Тикер", "Список наблюдения", "ИИ-чат"])

    with tab_ticker:
        st.subheader("Обзор тикера")
        default_symbol = (
            requested_symbol if requested_symbol in watchlist else (watchlist[0] if watchlist else "AAPL")
        )
        default_index = watchlist.index(default_symbol) if default_symbol in watchlist else 0
        left, right = st.columns([2, 1])
        selected_symbol = left.selectbox("Тикер из списка", options=watchlist, index=default_index)
        manual_symbol = right.text_input("Ручной ввод тикера", value=selected_symbol, max_chars=10)
        ticker = (manual_symbol or selected_symbol).upper()

        if st.session_state.get("backtest_last_symbol") != ticker:
            st.session_state["backtest_last_symbol"] = ticker
            reset_backtest_state(ticker)

        context = gather_ticker_context(ticker, active_provider)
        st.session_state["chat_ticker"] = ticker
        st.session_state["chat_context"] = context
        st.session_state["chat_provider"] = active_provider
        quote_source_placeholder.markdown(
            f"<div class='sidebar-badge'><strong>Цена:</strong> {translate_source(context['quote_source'])}</div>",
            unsafe_allow_html=True,
        )
        history_source_placeholder.markdown(
            f"<div class='sidebar-badge'><strong>История:</strong> {translate_source(context['history_source'])}</div>",
            unsafe_allow_html=True,
        )
        news_source_placeholder.markdown(
            f"<div class='sidebar-badge'><strong>Новости:</strong> {translate_source(context['news_source'])}</div>",
            unsafe_allow_html=True,
        )

        quote = context["quote"]
        history: pd.DataFrame = context["history"]
        analysis = context["analysis"]

        render_kpi_cards(quote, history, context)

        price_chart = render_candles(history)
        st.plotly_chart(price_chart, use_container_width=True)

        highlights_col, side_col = st.columns([2, 1])
        render_highlights_section(highlights_col, analysis, context["news_source"])
        render_sentiment_section(side_col, analysis)

        st.markdown("### Мини-бэктест")
        bt_cols = st.columns([1, 1, 1])
        fast_period = int(bt_cols[0].number_input("Быстрая MA", min_value=3, max_value=60, value=10, step=1))
        slow_period = int(bt_cols[1].number_input("Медленная MA", min_value=5, max_value=200, value=30, step=1))
        run_backtest_key = f"run_backtest_{ticker}"
        if bt_cols[2].button("Запустить", key=run_backtest_key):
            result = run_backtest(history, fast_period, slow_period)
            st.session_state.setdefault("backtests", {})[ticker] = result
            st.session_state.setdefault("backtest_params", {})[ticker] = (fast_period, slow_period)
        stored_bt = st.session_state.setdefault("backtests", {}).get(ticker)
        params = st.session_state.setdefault("backtest_params", {}).get(ticker)
        if stored_bt is not None and not stored_bt.empty:
            bt_fig = go.Figure()
            bt_fig.add_trace(
                go.Scatter(x=stored_bt["time"], y=stored_bt["equity"], mode="lines", name="Капитал")
            )
            bt_fig.update_layout(
                margin=dict(l=40, r=20, t=30, b=40), yaxis_title="Стоимость портфеля"
            )
            st.plotly_chart(bt_fig, use_container_width=True)
            if params:
                st.caption(f"Параметры: быстрая={params[0]}, медленная={params[1]}")
        elif stored_bt is not None:
            st.info("Недостаточно данных для запуска теста.")

        payload = build_api_payload(context)
        with st.expander("Предпросмотр API"):
            st.json(payload)

        pct_value = "—"
        pct_raw = quote.get("pct")
        if isinstance(pct_raw, (int, float, np.floating)):
            pct_value = f"{pct_raw:+.2f}%"

        export_cols = st.columns(2)
        header_info = {
            "Тикер": ticker,
            "Цена": format_price(quote.get("price")),
            "Δ%": pct_value,
            "Тональность": analysis.get("sentiment", "Нейтральная"),
        }
        try:
            png_bytes = export_png(price_chart, header_info)
            export_cols[0].download_button(
                "Скачать PNG",
                data=png_bytes,
                file_name=f"{ticker}_card.png",
                mime="image/png",
            )
        except RuntimeError:
            export_cols[0].warning(
                "PNG экспорт недоступен: установите пакет `kaleido` (например, `pip install -U kaleido`)."
            )
        html_data = export_html_card(context)
        export_cols[1].download_button(
            "Скачать HTML-карту",
            data=html_data,
            file_name=f"{ticker}_card.html",
            mime="text/html",
        )

    with tab_chat:
        st.subheader("ИИ-чат")
        chat_ticker = st.session_state.get("chat_ticker") or (
            requested_symbol if requested_symbol else (watchlist[0] if watchlist else "AAPL")
        )
        chat_context = st.session_state.get("chat_context", {})
        chat_provider = st.session_state.get("chat_provider", active_provider)
        st.session_state["chat_provider"] = chat_provider

        if not chat_context or chat_context.get("ticker") != chat_ticker:
            chat_context = gather_ticker_context(chat_ticker, chat_provider)
            st.session_state["chat_context"] = chat_context

        status_cols = st.columns([1, 1, 2])
        status_cols[0].markdown(f"**Текущий тикер:** `{chat_ticker}`")
        if chat_provider == "ollama":
            if is_ollama_chat_available():
                status_cols[1].success("Ollama chat готов")
            elif is_ollama_available():
                status_cols[1].warning("Используем fallback generate")
            else:
                status_cols[1].error("Ollama недоступен")
        elif chat_provider == "openai":
            status_cols[1].info("Режим OpenAI")
        else:
            status_cols[1].warning("LLM отключён")

        if "chat_use_context" not in st.session_state:
            st.session_state["chat_use_context"] = bool(chat_context)
        use_context = st.checkbox("Подмешивать рыночный контекст", key="chat_use_context")
        if status_cols[2].button("Очистить диалог", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        chat_enabled = use_llm and chat_provider in {"openai", "ollama"}
        if not chat_enabled:
            st.info("Активируйте LLM в настройках слева, чтобы использовать чат.")

        history: List[Dict[str, str]] = st.session_state.setdefault("chat_history", [])
        trim_chat_history(history)

        for message in history:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

        prompt = st.chat_input(
            "Задайте вопрос об идеях, драйверах или рисках", disabled=not chat_enabled
        )
        if prompt and chat_enabled:
            with st.chat_message("user"):
                st.markdown(prompt)
            history.append({"role": "user", "content": prompt})
            trim_chat_history(history)

            system_prompt = build_system_prompt(chat_ticker, chat_context or {}, use_context)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history[-20:])

            with st.spinner("StockSense обрабатывает запрос..."):
                response = llm_chat_completion(messages, provider=chat_provider, max_tokens=450)

            if not response:
                response = "Не удалось получить ответ от модели. Попробуйте снова позже."

            history.append({"role": "assistant", "content": response})
            trim_chat_history(history)
            with st.chat_message("assistant"):
                st.markdown(response)

    with tab_watch:
        st.subheader("Список наблюдения")
        preset_label = st.selectbox(
            "Предустановки / фильтры", WATCHLIST_FILTER_OPTIONS, key="watchlist_filter"
        )
        if st.button("Обновить данные", key="refresh_watchlist"):
            requests_cache.clear()
            fetch_history.clear()
            pull_company_news.clear()
            compute_volatility.clear()
            st.rerun()

        filtered_symbols = filter_watchlist(watchlist, preset_label)
        table = render_watchlist_table(filtered_symbols)
        if not table.empty:
            st.dataframe(table, use_container_width=True)
        else:
            st.info("Нет тикеров для отображения.")

        st.markdown("### Управление списком")
        add_cols = st.columns([3, 1])
        new_symbol = add_cols[0].text_input(
            "Добавить тикер", value="", placeholder="Например, SNOW", key="add_symbol"
        )
        if add_cols[1].button("Добавить", key="add_button"):
            candidate = new_symbol.strip().upper()
            if candidate and candidate not in watchlist:
                watchlist.append(candidate)
                st.session_state.watchlist = watchlist
                save_watchlist(watchlist)
                st.rerun()

        st.markdown("### Удаление из списка")
        for symbol in watchlist[:]:
            cols = st.columns([5, 1])
            cols[0].markdown(f"**{symbol}**")
            if cols[1].button("Удалить", key=f"remove_{symbol}"):
                watchlist.remove(symbol)
                st.session_state.watchlist = watchlist
                save_watchlist(watchlist)
                st.rerun()


if __name__ == "__main__":
    main()
