"""
Checkpoint utility for sync mode processing with crash recovery and intermediate saving.
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.data_model import CategoryResult, TextItem
from .logging import get_logger

logger = get_logger(__name__)


class SyncCheckpointer:
    """Handles checkpointing and recovery for synchronous pipeline processing"""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        session_id: Optional[str] = None,
        checkpoint_interval: int = 100,  # Save checkpoint every N records
        auto_cleanup: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Generate unique session ID if not provided
        if session_id is None:
            session_id = f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_id = session_id

        self.checkpoint_interval = checkpoint_interval
        self.auto_cleanup = auto_cleanup

        # Checkpoint files
        self.state_file = self.checkpoint_dir / f"{self.session_id}_state.json"
        self.items_file = self.checkpoint_dir / f"{self.session_id}_items.pkl"
        self.results_file = self.checkpoint_dir / f"{self.session_id}_results.json"

        # State tracking
        self.processed_count = 0
        self.total_results = 0
        self.start_time = time.time()

        logger.info(f"Initialized checkpointer for session: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Will save checkpoint every {checkpoint_interval} records")

    def save_checkpoint(
        self,
        pending_items: List[TextItem],
        processed_count: int,
        total_results: int,
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save current pipeline state to checkpoint files"""
        try:
            # Save state information
            state = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "processed_count": processed_count,
                "total_results": total_results,
                "start_time": self.start_time,
                "pending_items_count": len(pending_items),
                "additional_state": additional_state or {},
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Save pending items (binary for efficiency)
            if pending_items:
                with open(self.items_file, "wb") as f:
                    pickle.dump(pending_items, f)
            elif self.items_file.exists():
                # Remove items file if no pending items
                self.items_file.unlink()

            # Update internal state
            self.processed_count = processed_count
            self.total_results = total_results

            logger.info(
                f"💾 Checkpoint saved: {processed_count} processed, {total_results} results"
            )

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - checkpointing failure shouldn't stop processing

    def save_intermediate_results(self, results: List[CategoryResult]) -> None:
        """Save intermediate results to checkpoint results file"""
        try:
            # Load existing results if file exists
            existing_results = []
            if self.results_file.exists():
                with open(self.results_file, "r") as f:
                    existing_results = json.load(f)

            # Convert new results to dict format
            new_results = [
                {
                    "id": result.id,
                    "categories": result.categories,
                    "confidence_scores": result.confidence_scores,
                    "model_response": result.model_response,
                    "processed_at": (
                        result.processed_at.isoformat() if result.processed_at else None
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                for result in results
            ]

            # Append new results
            existing_results.extend(new_results)

            # Save back to file
            with open(self.results_file, "w") as f:
                json.dump(existing_results, f, indent=2)

            logger.debug(f"Saved {len(new_results)} intermediate results")

        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def load_checkpoint(
        self,
    ) -> Tuple[Optional[List[TextItem]], Optional[Dict[str, Any]]]:
        """Load checkpoint state and return pending items and state info"""
        try:
            if not self.state_file.exists():
                logger.info("No checkpoint found for this session")
                return None, None

            # Load state
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Load pending items if they exist
            pending_items = None
            if self.items_file.exists():
                with open(self.items_file, "rb") as f:
                    pending_items = pickle.load(f)

            # Update internal state
            self.processed_count = state.get("processed_count", 0)
            self.total_results = state.get("total_results", 0)
            self.start_time = state.get("start_time", time.time())

            logger.info(
                f"Checkpoint loaded: {self.processed_count} processed, "
                f"{len(pending_items) if pending_items else 0} pending, "
                f"{self.total_results} results"
            )

            return pending_items, state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None, None

    def should_checkpoint(self, processed_count: int) -> bool:
        """Check if it's time to save a checkpoint"""
        return processed_count % self.checkpoint_interval == 0

    def get_intermediate_results(self) -> Optional[List[Dict[str, Any]]]:
        """Get intermediate results saved during processing"""
        try:
            if self.results_file.exists():
                with open(self.results_file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up checkpoint files after successful completion"""
        if not self.auto_cleanup:
            logger.info("Auto-cleanup disabled, keeping checkpoint files")
            return

        try:
            files_to_remove = [self.state_file, self.items_file, self.results_file]
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed checkpoint file: {file_path}")

            logger.info(f"Cleaned up checkpoint files for session: {self.session_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint files: {e}")

    def list_available_sessions(self) -> List[str]:
        """List all available checkpoint sessions"""
        try:
            sessions = set()
            for file_path in self.checkpoint_dir.glob("*_state.json"):
                session_id = file_path.stem.replace("_state", "")
                sessions.add(session_id)

            return sorted(list(sessions))

        except Exception as e:
            logger.error(f"Failed to list checkpoint sessions: {e}")
            return []

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific checkpoint session"""
        try:
            state_file = self.checkpoint_dir / f"{session_id}_state.json"
            if not state_file.exists():
                return None

            with open(state_file, "r") as f:
                state = json.load(f)

            # Add file sizes and existence info
            items_file = self.checkpoint_dir / f"{session_id}_items.pkl"
            results_file = self.checkpoint_dir / f"{session_id}_results.json"

            state["files"] = {
                "state_file": str(state_file),
                "items_file": str(items_file) if items_file.exists() else None,
                "results_file": str(results_file) if results_file.exists() else None,
            }

            return state

        except Exception as e:
            logger.error(f"Failed to get session info for {session_id}: {e}")
            return None

    def estimate_progress(self, total_items: Optional[int] = None) -> Dict[str, Any]:
        """Estimate progress and completion time"""
        elapsed_time = time.time() - self.start_time

        progress_info = {
            "processed_count": self.processed_count,
            "total_results": self.total_results,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": f"{elapsed_time/60:.1f} minutes",
        }

        if total_items and self.processed_count > 0:
            progress_ratio = self.processed_count / total_items
            estimated_total_time = elapsed_time / progress_ratio
            remaining_time = estimated_total_time - elapsed_time

            progress_info.update(
                {
                    "total_items": total_items,
                    "progress_percent": progress_ratio * 100,
                    "estimated_remaining_seconds": remaining_time,
                    "estimated_remaining_formatted": f"{remaining_time/60:.1f} minutes",
                    "estimated_completion": datetime.fromtimestamp(
                        time.time() + remaining_time
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return progress_info
