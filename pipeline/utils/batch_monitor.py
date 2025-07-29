"""
Batch monitoring utilities for the NDE pipeline.

This module provides utilities for monitoring OpenAI batch jobs, including
intelligent polling, progress tracking, and completion waiting.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..llm.openai_provider import OpenAIProvider
from .logging import get_logger

logger = get_logger(__name__)


class BatchMonitor:
    """Monitor and manage OpenAI batch jobs with intelligent polling"""

    def __init__(self, provider: OpenAIProvider, check_interval_minutes: int = 30):
        self.provider = provider
        self.check_interval_minutes = check_interval_minutes
        self.check_interval_seconds = check_interval_minutes * 60

    async def wait_for_completion_with_monitoring(
        self, batch_ids: List[str], progress_callback: Optional[Callable] = None
    ) -> Dict[str, List]:
        """
        Wait for batch completion with periodic monitoring and logging

        Args:
            batch_ids: List of batch IDs to monitor
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary mapping batch_id to results
        """
        logger.info(f"Starting monitoring for {len(batch_ids)} batches")
        logger.info(f"Will check every {self.check_interval_minutes} minutes")

        completed_batches = {}
        failed_batches = set()
        monitoring_start = datetime.now()

        # Validate batch IDs first
        valid_ids, invalid_ids = await self.provider.validate_batch_ids(batch_ids)

        if invalid_ids:
            logger.error(f"Invalid batch IDs found: {invalid_ids}")

        if not valid_ids:
            logger.error("No valid batch IDs to monitor")
            return {}

        logger.info(f"Monitoring {len(valid_ids)} valid batches")

        iteration = 0
        while len(completed_batches) + len(failed_batches) < len(valid_ids):
            iteration += 1
            check_time = datetime.now()
            elapsed_total = check_time - monitoring_start

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Batch Check #{iteration} - {check_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.info(f"Total monitoring time: {elapsed_total}")
            logger.info(
                f"Completed: {len(completed_batches)}, Failed: {len(failed_batches)}, Remaining: {len(valid_ids) - len(completed_batches) - len(failed_batches)}"
            )
            logger.info(f"{'='*60}")

            for batch_id in valid_ids:
                if batch_id in completed_batches or batch_id in failed_batches:
                    continue

                try:
                    details = await self.provider.get_batch_status_details(batch_id)
                    if not details:
                        logger.error(f"Could not get status for batch {batch_id}")
                        continue

                    status = details["status"]
                    elapsed = details["elapsed_time"]
                    eta = details.get("estimated_completion", "Unknown")

                    logger.info(
                        f"Batch {batch_id}: {status} (elapsed: {elapsed}, ETA: {eta})"
                    )

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(batch_id, details)

                    if status == "completed":
                        logger.info(
                            f"✓ Batch {batch_id} completed! Retrieving results..."
                        )
                        results = await self.provider.get_batch_results(batch_id)
                        if results:
                            completed_batches[batch_id] = results
                            logger.info(
                                f"✓ Retrieved {len(results)} results from batch {batch_id}"
                            )
                        else:
                            logger.error(
                                f"✗ Failed to retrieve results from completed batch {batch_id}"
                            )
                            failed_batches.add(batch_id)

                    elif status in ["failed", "cancelled", "expired"]:
                        logger.error(f"✗ Batch {batch_id} {status}")
                        failed_batches.add(batch_id)

                    else:
                        # Still in progress - log additional details if available
                        if details.get("request_counts"):
                            counts = details["request_counts"]
                            logger.info(f"  Request counts: {counts}")

                except Exception as e:
                    logger.error(f"Error checking batch {batch_id}: {e}")
                    await asyncio.sleep(10)  # Short delay before continuing

            # Check if we're done
            remaining = len(valid_ids) - len(completed_batches) - len(failed_batches)
            if remaining == 0:
                break

            # Wait before next check
            logger.info(
                f"\nWaiting {self.check_interval_minutes} minutes before next check..."
            )
            logger.info(
                f"Next check at: {(datetime.now() + timedelta(minutes=self.check_interval_minutes)).strftime('%Y-%m-%d %H:%M:%S')}"
            )

            await asyncio.sleep(self.check_interval_seconds)

        # Final summary
        total_monitoring_time = datetime.now() - monitoring_start
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH MONITORING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total monitoring time: {total_monitoring_time}")
        logger.info(f"Completed batches: {len(completed_batches)}")
        logger.info(f"Failed batches: {len(failed_batches)}")

        if failed_batches:
            logger.warning(f"Failed batch IDs: {list(failed_batches)}")

        return completed_batches

    async def monitor_single_batch(
        self, batch_id: str, progress_callback: Optional[Callable] = None
    ) -> Optional[List]:
        """Monitor a single batch until completion"""
        results = await self.wait_for_completion_with_monitoring(
            [batch_id], progress_callback
        )
        return results.get(batch_id)

    def create_progress_logger(
        self, log_to_file: bool = True, log_file: Optional[Path] = None
    ) -> Callable:
        """Create a progress callback that logs to file and console"""
        if log_to_file:
            if not log_file:
                log_file = (
                    Path("logs")
                    / f"batch_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            log_file.parent.mkdir(exist_ok=True)

        def progress_callback(batch_id: str, details: Dict):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = details["status"]
            elapsed = details["elapsed_time"]
            eta = details.get("estimated_completion", "Unknown")

            log_message = f"[{timestamp}] Batch {batch_id}: {status} (elapsed: {elapsed}, ETA: {eta})"

            # Log to console (already handled by logger)
            logger.info(f"Progress: {log_message}")

            # Log to file if requested
            if log_to_file:
                with open(log_file, "a") as f:
                    f.write(log_message + "\n")

        return progress_callback
