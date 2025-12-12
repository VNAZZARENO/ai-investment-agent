"""
Token usage tracking and cost estimation module.
Provides comprehensive logging of LLM token consumption across all agents.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = structlog.get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage data for a single LLM call."""
    timestamp: str
    agent_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost in USD assuming paid tier (GCP project with billing enabled).

        IMPORTANT: If your GCP project has billing enabled, ALL API calls cost money,
        regardless of model name or "free tier" marketing. These are paid tier rates.

        Updated for Dec 2025 pricing (sources: ai.google.dev/gemini-api/docs/pricing).
        """
        # LLM pricing (per 1M tokens)
        # IMPORTANT: Order matters! More specific models must come before general ones
        pricing = {
            # ============================================================
            # Claude/Anthropic Models (Dec 2025 pricing)
            # ============================================================
            "claude-opus-4-5": {
                "prompt": 5.00,      # $5.00 per 1M input tokens
                "completion": 25.00  # $25.00 per 1M output tokens
            },
            "claude-sonnet-4-5": {
                "prompt": 3.00,      # $3.00 per 1M input tokens
                "completion": 15.00  # $15.00 per 1M output tokens
            },
            "claude-haiku-4-5": {
                "prompt": 1.00,      # $1.00 per 1M input tokens
                "completion": 5.00   # $5.00 per 1M output tokens
            },
            # Legacy Claude models
            "claude-sonnet-4": {
                "prompt": 3.00,
                "completion": 15.00
            },
            "claude-3-5-haiku": {
                "prompt": 0.80,
                "completion": 4.00
            },
            # ============================================================
            # Gemini Models (Dec 2025 pricing)
            # ============================================================
            # Gemini 2.0 Flash variants
            "gemini-2.0-flash-thinking-exp": {
                "prompt": 0.30,     # Paid tier: $0.30 per 1M input tokens
                "completion": 2.50  # Paid tier: $2.50 per 1M output tokens
            },
            "gemini-2.0-flash-exp": {
                "prompt": 0.30,
                "completion": 2.50
            },
            # Gemini 2.5 Flash variants
            "gemini-2.5-flash-lite": {
                "prompt": 0.10,     # $0.10 per 1M input tokens
                "completion": 0.40  # $0.40 per 1M output tokens
            },
            "gemini-2.5-flash": {
                "prompt": 0.30,     # $0.30 per 1M input tokens
                "completion": 2.50  # $2.50 per 1M output tokens
            },
            # Gemini 3 Pro variants
            "gemini-3-pro-preview": {
                "prompt": 2.00,     # $2.00 per 1M input tokens
                "completion": 12.00 # $12.00 per 1M output tokens
            },
            "gemini-3-pro": {
                "prompt": 2.00,
                "completion": 12.00
            },
        }

        # Default pricing for unknown models (assume Flash-level pricing)
        default_pricing = {"prompt": 0.30, "completion": 2.50}

        # Find pricing for this model (match by prefix)
        model_pricing = default_pricing
        for model_key, prices in pricing.items():
            if self.model_name.startswith(model_key):
                model_pricing = prices
                break

        prompt_cost = (self.prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (self.completion_tokens / 1_000_000) * model_pricing["completion"]

        return prompt_cost + completion_cost


@dataclass
class AgentTokenStats:
    """Aggregate token statistics for a single agent."""
    agent_name: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: List[TokenUsage] = field(default_factory=list)

    def add_usage(self, usage: TokenUsage):
        """Add a token usage record to this agent's stats."""
        self.calls.append(usage)
        self.total_calls += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost_usd += usage.estimated_cost_usd


class TokenTracker:
    """
    Global token tracker that aggregates usage across all agents.
    Thread-safe singleton for tracking LLM token consumption.
    """

    _instance: Optional['TokenTracker'] = None
    _quiet_mode: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.agent_stats: Dict[str, AgentTokenStats] = {}
        self.all_usages: List[TokenUsage] = []
        self.session_start = datetime.now().isoformat()

        # Only log if quiet mode is not enabled (check both class and instance level)
        if not self.__class__._quiet_mode and not self._quiet_mode:
            logger.info("token_tracker_initialized", session_start=self.session_start)

    @classmethod
    def set_quiet_mode(cls, quiet: bool = True):
        """Enable or disable quiet mode to suppress logging."""
        cls._quiet_mode = quiet

    def record_usage(
        self,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Record token usage for a specific agent."""
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

        # Add to agent-specific stats
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = AgentTokenStats(agent_name=agent_name)

        self.agent_stats[agent_name].add_usage(usage)
        self.all_usages.append(usage)

        if not self._quiet_mode:
            logger.info(
                "token_usage_recorded",
                agent=agent_name,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=f"${usage.estimated_cost_usd:.6f}"
            )

    def get_agent_stats(self, agent_name: str) -> Optional[AgentTokenStats]:
        """Get statistics for a specific agent."""
        return self.agent_stats.get(agent_name)

    def get_total_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all agents."""
        total_prompt = sum(stats.total_prompt_tokens for stats in self.agent_stats.values())
        total_completion = sum(stats.total_completion_tokens for stats in self.agent_stats.values())
        total_cost = sum(stats.total_cost_usd for stats in self.agent_stats.values())

        return {
            "total_calls": len(self.all_usages),
            "total_agents": len(self.agent_stats),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": total_cost,
            "session_start": self.session_start,
            "agents": {
                name: {
                    "calls": stats.total_calls,
                    "prompt_tokens": stats.total_prompt_tokens,
                    "completion_tokens": stats.total_completion_tokens,
                    "total_tokens": stats.total_tokens,
                    "cost_usd": stats.total_cost_usd
                }
                for name, stats in self.agent_stats.items()
            }
        }

    def reset(self):
        """Reset all tracking data (useful for new analysis runs)."""
        self.agent_stats.clear()
        self.all_usages.clear()
        self.session_start = datetime.now().isoformat()
        if not self._quiet_mode:
            logger.info("token_tracker_reset", session_start=self.session_start)

    def print_summary(self):
        """Print a formatted summary of token usage to logger."""
        if self._quiet_mode:
            return

        stats = self.get_total_stats()

        logger.info(
            "=" * 80 + "\n" +
            "TOKEN USAGE SUMMARY\n" +
            "=" * 80
        )
        logger.info(f"Session Start: {stats['session_start']}")
        logger.info(f"Total LLM Calls: {stats['total_calls']}")
        logger.info(f"Total Agents: {stats['total_agents']}")
        logger.info(f"Total Prompt Tokens: {stats['total_prompt_tokens']:,}")
        logger.info(f"Total Completion Tokens: {stats['total_completion_tokens']:,}")
        logger.info(f"Total Tokens: {stats['total_tokens']:,}")
        logger.info(f"Projected Cost (Paid Tier): ${stats['total_cost_usd']:.4f} USD")
        logger.info("  (Note: Actual cost = $0 if using free tier without billing enabled)")
        logger.info("\nPer-Agent Breakdown:")
        logger.info("-" * 80)

        # Sort agents by cost (descending)
        sorted_agents = sorted(
            stats['agents'].items(),
            key=lambda x: x[1]['cost_usd'],
            reverse=True
        )

        for agent_name, agent_stats in sorted_agents:
            logger.info(
                f"\n{agent_name}:\n"
                f"  Calls: {agent_stats['calls']}\n"
                f"  Prompt Tokens: {agent_stats['prompt_tokens']:,}\n"
                f"  Completion Tokens: {agent_stats['completion_tokens']:,}\n"
                f"  Total Tokens: {agent_stats['total_tokens']:,}\n"
                f"  Cost: ${agent_stats['cost_usd']:.4f}"
            )

        logger.info("=" * 80)


class TokenTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback handler that tracks token usage per agent.
    Attach this to LLM instances to automatically log token consumption.
    """

    def __init__(self, agent_name: str, tracker: Optional[TokenTracker] = None):
        """
        Initialize callback with agent name.

        Args:
            agent_name: Name of the agent using this LLM
            tracker: Optional TokenTracker instance (uses singleton if not provided)
        """
        super().__init__()
        self.agent_name = agent_name
        self.tracker = tracker or TokenTracker()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM completes a generation."""
        try:
            # Extract token usage from response
            # Supports both Gemini and Claude/Anthropic response formats
            usage_metadata = {}
            model_name = "unknown"

            # Try to get usage from generations first
            if response.generations and len(response.generations) > 0:
                first_generation_list = response.generations[0]
                if first_generation_list and len(first_generation_list) > 0:
                    first_generation = first_generation_list[0]

                    # Check if it's an AIMessage with usage_metadata (Gemini)
                    if hasattr(first_generation, 'message') and hasattr(first_generation.message, 'usage_metadata'):
                        meta = first_generation.message.usage_metadata
                        if meta:
                            # Handle both dict and object attribute access
                            if isinstance(meta, dict):
                                usage_metadata = meta
                            else:
                                # Claude returns usage_metadata as an object with attributes
                                usage_metadata = {
                                    'input_tokens': getattr(meta, 'input_tokens', 0),
                                    'output_tokens': getattr(meta, 'output_tokens', 0),
                                }

                    # Check response_metadata for Claude/Anthropic
                    if hasattr(first_generation, 'message') and hasattr(first_generation.message, 'response_metadata'):
                        resp_meta = first_generation.message.response_metadata
                        if resp_meta and isinstance(resp_meta, dict):
                            # Claude puts usage in response_metadata.usage
                            if 'usage' in resp_meta:
                                claude_usage = resp_meta['usage']
                                if isinstance(claude_usage, dict):
                                    usage_metadata = {
                                        'input_tokens': claude_usage.get('input_tokens', 0),
                                        'output_tokens': claude_usage.get('output_tokens', 0),
                                    }
                            # Get model name
                            if model_name == "unknown":
                                model_name = resp_meta.get('model', resp_meta.get('model_name', 'unknown'))

                    # Get model name from generation_info (Gemini)
                    if model_name == "unknown" and hasattr(first_generation, 'generation_info'):
                        gen_info = first_generation.generation_info
                        if gen_info and isinstance(gen_info, dict):
                            model_name = gen_info.get('model_name', 'unknown')

            # Fallback to llm_output (for other LLM providers)
            if not usage_metadata and response.llm_output:
                if isinstance(response.llm_output, dict):
                    usage_metadata = response.llm_output.get("usage_metadata", {})
                    if not usage_metadata:
                        usage_metadata = response.llm_output.get("token_usage", {})
                    if model_name == "unknown":
                        model_name = response.llm_output.get("model_name", "unknown")

            if usage_metadata:
                prompt_tokens = usage_metadata.get("input_tokens", 0) or usage_metadata.get("prompt_tokens", 0)
                completion_tokens = usage_metadata.get("output_tokens", 0) or usage_metadata.get("completion_tokens", 0)

                if prompt_tokens > 0 or completion_tokens > 0:
                    self.tracker.record_usage(
                        agent_name=self.agent_name,
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
        except Exception as e:
            # Don't let token tracking errors break the analysis
            logger.debug(f"Token tracking error (non-fatal): {e}")


# Global singleton instance (lazy initialization to respect quiet mode)
_global_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """Get the global TokenTracker singleton (lazy initialization)."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker
