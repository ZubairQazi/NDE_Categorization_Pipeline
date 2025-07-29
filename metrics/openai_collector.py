import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import tiktoken

from .base_collector import BaseMetricsCollector

logger = logging.getLogger(__name__)


class OpenAIMetricsCollector(BaseMetricsCollector):
    def __init__(self, model: str, output_dir: str = "metrics/metrics_output"):
        super().__init__(output_dir)
        self.model = model

        # Initialize tiktoken for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(
                f"Model {model} not found in tiktoken, falling back to o200k_base encoding"
            )
            self.encoding = tiktoken.get_encoding("o200k_base")

        # Simple metrics tracking for the requested stats
        self.metrics.update(
            {
                "peak_memory_mb": 0.0,
                "total_memory_samples": 0,
                "memory_sum": 0.0,
                "peak_cpu_percent": 0.0,
                "total_cpu_samples": 0,
                "cpu_sum": 0.0,
            }
        )

        # OpenAI pricing per 1M tokens
        self.pricing = {
            # GPT-4o models
            "gpt-4o": {
                "batch": {"input": 1.25, "output": 5.00},
                "non_batch": {"input": 2.50, "output": 10.00},
            },
            "gpt-4o-mini": {
                "batch": {"input": 0.075, "output": 0.30},
                "non_batch": {"input": 0.15, "output": 0.60},
            },
            # GPT-4.1 models
            "gpt-4.1": {
                "batch": {"input": 1.00, "output": 4.00},
                "non_batch": {"input": 2.00, "output": 8.00},
            },
            "gpt-4.1-nano": {
                "batch": {"input": 0.05, "output": 0.20},
                "non_batch": {"input": 0.10, "output": 0.40},
            },
        }

        # Get pricing for the model, default to gpt-4o-mini if unknown
        default_pricing = self.pricing.get("gpt-4o-mini")
        model_pricing = self.pricing.get(model, default_pricing)

        # Initialize with batch pricing by default
        self.input_price = model_pricing["batch"]["input"]
        self.output_price = model_pricing["batch"]["output"]
        self._model_pricing = model_pricing

    def set_pricing_mode(self, is_batch: bool):
        """Set pricing mode to batch or non-batch"""
        mode = "batch" if is_batch else "non_batch"
        self.input_price = self._model_pricing[mode]["input"]
        self.output_price = self._model_pricing[mode]["output"]

    def start_collection(self):
        """Start collecting metrics"""
        self.metrics["start_time"] = datetime.now()

    def stop_collection(self):
        """Stop collecting metrics and calculate final metrics"""
        self.metrics["end_time"] = datetime.now()

        # Calculate total runtime
        if self.metrics["start_time"] and self.metrics["end_time"]:
            self.metrics["total_runtime"] = (
                self.metrics["end_time"] - self.metrics["start_time"]
            ).total_seconds()

        # Calculate average time per sample
        if self.metrics["total_samples"] > 0:
            self.metrics["average_time_per_sample"] = (
                self.metrics["total_runtime"] / self.metrics["total_samples"]
            )

    def calculate_token_cost(self, prompt: str, completion: str) -> float:
        """Calculate the cost for a single request based on tokens"""
        prompt_tokens = len(self.encoding.encode(prompt))
        completion_tokens = len(self.encoding.encode(completion))

        # Convert to millions of tokens for pricing
        input_cost = (prompt_tokens / 1_000_000) * self.input_price
        output_cost = (completion_tokens / 1_000_000) * self.output_price

        return input_cost + output_cost

    def calculate_batch_cost(self, prompts: List[str], completions: List[str]) -> float:
        """Calculate the total cost for a batch of requests"""
        total_input_tokens = sum(len(self.encoding.encode(p)) for p in prompts)
        total_output_tokens = sum(len(self.encoding.encode(c)) for c in completions)

        # Convert to millions of tokens for pricing
        input_cost = (total_input_tokens / 1_000_000) * self.input_price
        output_cost = (total_output_tokens / 1_000_000) * self.output_price

        return input_cost + output_cost

    def record_request(
        self,
        request_id: str,
        start_time: datetime,
        end_time: datetime,
        prompt: str,
        completion: str,
        error: Optional[str] = None,
    ):
        """Record metrics for a single synchronous request"""
        # Set non-batch pricing for single requests
        self.set_pricing_mode(is_batch=False)

        cost = self.calculate_token_cost(prompt, completion)
        prompt_tokens = len(self.encoding.encode(prompt))
        completion_tokens = len(self.encoding.encode(completion))

        request_metrics = {
            "request_id": request_id,
            "start_time": start_time,
            "end_time": end_time,
            "runtime": (end_time - start_time).total_seconds(),
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
            "pricing_mode": "non_batch",
        }

        self.metrics["batch_metrics"].append(request_metrics)
        self.metrics["total_samples"] += 1
        self.metrics["total_cost"] += cost

        if error:
            self.metrics["errors"].append(
                {"request_id": request_id, "error": error, "timestamp": datetime.now()}
            )

        # Update cost per sample
        if self.metrics["total_samples"] > 0:
            self.metrics["cost_per_sample"] = (
                self.metrics["total_cost"] / self.metrics["total_samples"]
            )

    def record_batch(
        self,
        batch_id: str,
        num_samples: int,
        start_time: datetime,
        end_time: datetime,
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        cost: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Record metrics for a batch of requests"""
        # Set batch pricing for batch requests
        self.set_pricing_mode(is_batch=True)

        # Calculate cost if prompts and completions are provided
        if prompts and completions and cost is None:
            cost = self.calculate_batch_cost(prompts, completions)

        batch_metrics = {
            "batch_id": batch_id,
            "num_samples": num_samples,
            "start_time": start_time,
            "end_time": end_time,
            "runtime": (end_time - start_time).total_seconds(),
            "cost": cost or 0.0,
            "error": error,
            "pricing_mode": "batch",
        }

        # Add token counts if available
        if prompts and completions:
            total_input_tokens = sum(len(self.encoding.encode(p)) for p in prompts)
            total_output_tokens = sum(len(self.encoding.encode(c)) for c in completions)
            batch_metrics.update(
                {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "average_input_tokens": total_input_tokens / num_samples,
                    "average_output_tokens": total_output_tokens / num_samples,
                }
            )

        self.metrics["batch_metrics"].append(batch_metrics)
        self.metrics["total_samples"] += num_samples
        self.metrics["total_cost"] += cost or 0.0

        if error:
            self.metrics["errors"].append(
                {"batch_id": batch_id, "error": error, "timestamp": datetime.now()}
            )

        # Update cost per sample
        if self.metrics["total_samples"] > 0:
            self.metrics["cost_per_sample"] = (
                self.metrics["total_cost"] / self.metrics["total_samples"]
            )

    def record_resource_usage(
        self,
        cpu_usage: float,
        memory_usage: float,
        timestamp: Optional[datetime] = None,
    ):
        """Record system resource usage metrics for API-based models"""
        resource_metrics = {
            "timestamp": timestamp or datetime.now(),
            "cpu_percent": cpu_usage,
            "memory_used_mb": memory_usage,
        }
        self.metrics["resource_usage"].append(resource_metrics)

        # Update peak and running averages for the simple stats
        self.metrics["peak_cpu_percent"] = max(
            self.metrics["peak_cpu_percent"], cpu_usage
        )
        self.metrics["peak_memory_mb"] = max(
            self.metrics["peak_memory_mb"], memory_usage
        )

        self.metrics["cpu_sum"] += cpu_usage
        self.metrics["memory_sum"] += memory_usage
        self.metrics["total_cpu_samples"] += 1
        self.metrics["total_memory_samples"] += 1

    def get_simple_stats(self) -> Dict[str, Any]:
        """Get the specific stats requested: Model, Total Runtime, Time per sample, Peak Memory, Average Memory, Peak CPU, Average CPU, Total Cost, Cost per Sample"""

        # Calculate averages
        avg_memory = self.metrics["memory_sum"] / max(
            self.metrics["total_memory_samples"], 1
        )
        avg_cpu = self.metrics["cpu_sum"] / max(self.metrics["total_cpu_samples"], 1)

        return {
            "Model": self.model,
            "Total Runtime": f"{self.metrics.get('total_runtime', 0):.2f}s",
            "Time per sample": f"{self.metrics.get('average_time_per_sample', 0):.3f}s",
            "Peak Memory": f"{self.metrics['peak_memory_mb']:.1f} MB",
            "Average Memory": f"{avg_memory:.1f} MB",
            "Peak CPU Usage": f"{self.metrics['peak_cpu_percent']:.1f}%",
            "Average CPU Usage": f"{avg_cpu:.1f}%",
            "Total Cost": f"${self.metrics.get('total_cost', 0):.4f}",
            "Cost per Sample": f"${self.metrics.get('cost_per_sample', 0):.4f}",
        }
