"""
LLM configuration and initialization module.
Supports multiple providers: Google Gemini and Anthropic Claude.
Includes token tracking for cost monitoring.
Provider selection via LLM_PROVIDER environment variable ("google" or "anthropic").
"""

import logging
from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import BaseCallbackHandler
from src.config import config

logger = logging.getLogger(__name__)

# Relax safety settings slightly for financial/market analysis context
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Configurable rate limiter based on Gemini API tier
# Tier detection via GEMINI_RPM_LIMIT environment variable:
#   - Free tier: 15 RPM (default)
#   - Paid tier 1: 360 RPM (set GEMINI_RPM_LIMIT=360)
#   - Paid tier 2: 1000 RPM (set GEMINI_RPM_LIMIT=1000)
#
# Rate limiter settings are calculated to be conservative:
# - RPS = RPM / 60 (convert to requests per second)
# - Reduce by 20% for safety margin to avoid hitting limits
# - max_bucket_size allows brief bursts without throttling

def _create_rate_limiter_from_rpm(rpm: int) -> InMemoryRateLimiter:
    """
    Create a rate limiter from RPM (requests per minute) setting.

    Args:
        rpm: Target requests per minute (e.g., 15 for free tier, 360 for paid)

    Returns:
        Configured InMemoryRateLimiter
    """
    # Convert RPM to RPS with 20% safety margin
    safety_factor = 0.8  # Use 80% of limit to avoid edge cases
    rps = (rpm / 60.0) * safety_factor

    # Bucket size: allow bursts up to 10% of RPM for parallel agent execution
    max_bucket = max(5, int(rpm * 0.1))

    logger.info(
        f"Rate limiter configured: {rpm} RPM → {rps:.2f} RPS "
        f"(80% of limit, bucket size: {max_bucket})"
    )

    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket
    )

# Initialize global rate limiter from config
GLOBAL_RATE_LIMITER = _create_rate_limiter_from_rpm(config.gemini_rpm_limit)

def create_gemini_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """Generic factory for Gemini models with optional callbacks."""

    # Note: transport='rest' is sometimes more stable than grpc for large contexts on some networks
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        # Increased timeout is handled by the caller (config.api_timeout)
        timeout=timeout,
        # Retry logic handles 500, 503, 504 automatically by default in LangChain
        max_retries=max_retries,
        safety_settings=SAFETY_SETTINGS,
        streaming=streaming,
        rate_limiter=GLOBAL_RATE_LIMITER,
        convert_system_message_to_human=False,
        # EXPLICITLY set max_output_tokens to prevent truncation
        max_output_tokens=32768,
        callbacks=callbacks or []
    )
    return llm


def create_claude_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """Generic factory for Claude/Anthropic models with optional callbacks."""

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        timeout=float(timeout),
        max_retries=max_retries,
        streaming=streaming,
        # Claude models have high token limits, set reasonable max for analysis
        max_tokens=8192,
        callbacks=callbacks or []
    )
    return llm


def create_model(
    model_name: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create LLM based on configured provider.

    Uses LLM_PROVIDER from config to determine which backend to use:
    - "google" (default): Google Gemini
    - "anthropic": Anthropic Claude
    """
    if config.llm_provider == "anthropic":
        return create_claude_model(
            model_name, temperature, timeout, max_retries, streaming, callbacks
        )
    else:
        return create_gemini_model(
            model_name, temperature, timeout, max_retries, streaming, callbacks
        )

def create_quick_thinking_llm(
    temperature: float = 0.3,
    model: Optional[str] = None,
    timeout: int = None, # Allow override or use config default
    max_retries: int = None, # Allow override or use config default
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create a quick thinking LLM.

    Provider-aware: uses Gemini 2.5 Flash or Claude Haiku 4.5 based on LLM_PROVIDER.
    """
    model_name = model or config.quick_think_llm
    # Use config defaults if not provided
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    logger.info(f"Initializing Quick LLM ({config.llm_provider}): {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_model(model_name, temperature, final_timeout, final_retries, callbacks=callbacks)


def create_deep_thinking_llm(
    temperature: float = 0.1,
    model: Optional[str] = None,
    timeout: int = None, # Allow override or use config default
    max_retries: int = None, # Allow override or use config default
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Create a deep thinking LLM.

    Provider-aware: uses Gemini 3 Pro or Claude Sonnet 4.5 based on LLM_PROVIDER.
    """
    model_name = model or config.deep_think_llm
    # Use config defaults if not provided
    final_timeout = timeout if timeout is not None else config.api_timeout
    final_retries = max_retries if max_retries is not None else config.api_retry_attempts

    logger.info(f"Initializing Deep LLM ({config.llm_provider}): {model_name} (timeout={final_timeout}, retries={final_retries})")
    return create_model(model_name, temperature, final_timeout, final_retries, callbacks=callbacks)

# Initialize default instances
quick_thinking_llm = create_quick_thinking_llm()
deep_thinking_llm = create_deep_thinking_llm()