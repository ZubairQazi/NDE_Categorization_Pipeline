# core/pipeline.py
import asyncio
from datetime import datetime
from typing import List, Literal, Optional, Type

from ..input.base import DataInput
from ..llm.base import LLMProvider
from ..output.base import DataOutput
from ..processors.base import DataProcessor
from ..utils.checkpoint import SyncCheckpointer
from ..utils.logging import get_logger
from .data_model import BatchJob, CategoryResult, JobStatus, TextItem

logger = get_logger(__name__)


class Pipeline:
    def __init__(
        self,
        input_handler: DataInput,
        llm_provider: LLMProvider,
        output_handler: DataOutput,
        preprocessors: Optional[List[DataProcessor]] = None,
        postprocessors: Optional[List[DataProcessor]] = None,
        categories: List[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 5,
        mode: Literal["sync", "batch"] = "sync",
        # Sync mode specific options
        enable_checkpointing: bool = True,
        checkpoint_interval: int = 10,
        session_id: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ):
        self.input_handler = input_handler
        self.llm_provider = llm_provider
        self.output_handler = output_handler
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.categories = categories
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mode = mode
        self.active_jobs: List[BatchJob] = []

        # Sync mode enhancements
        self.enable_checkpointing = enable_checkpointing and mode == "sync"
        self.checkpoint_interval = checkpoint_interval
        self.session_id = session_id
        self.resume_from_checkpoint = resume_from_checkpoint

        # Initialize checkpointer for sync mode
        if self.enable_checkpointing:
            self.checkpointer = SyncCheckpointer(
                session_id=session_id,
                checkpoint_interval=checkpoint_interval,
                auto_cleanup=True,
            )
        else:
            self.checkpointer = None

    def _process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Apply all input processors to the data"""
        processed_items = items
        for processor in self.preprocessors:
            processed_items = processor.process_input(processed_items)
        return processed_items

    def _process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Apply all output processors to the results"""
        processed_results = results
        for processor in self.postprocessors:
            processed_results = processor.process_output(processed_results)
        return processed_results

    async def process_items(
        self, items: List[TextItem]
    ) -> Optional[List[CategoryResult]]:
        """Process items based on selected mode"""
        processed_items = self._process_input(items)

        # Print items that were filtered out during processing
        filtered_ids = set(item.id for item in items) - set(
            item.id for item in processed_items
        )
        if filtered_ids:
            logger.info(f"Items filtered during processing: {filtered_ids}")
            for item in items:
                if item.id in filtered_ids:
                    logger.info(f"Filtered item {item.id}: {item.text}")

        for attempt in range(self.max_retries):
            try:
                if self.mode == "sync":
                    # Process synchronously
                    results = self.llm_provider.categorize(
                        processed_items, self.categories
                    )
                else:
                    # Process as batch with monitoring
                    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logger.info(f"Submitting batch: {batch_name}")
                    results = await self.llm_provider.batch_categorize(
                        processed_items, self.categories, batch_name
                    )
                    logger.info(
                        f"Batch processing completed, received {len(results) if results else 0} results"
                    )

                return self._process_output(results)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to process items after {self.max_retries} attempts: {e}"
                    )
                    raise
                await asyncio.sleep(self.retry_delay)

    async def run(self) -> None:
        """Run the pipeline with enhanced error handling, progress tracking, and checkpointing"""
        # Topic categories are necessary for any post processing
        if not self.categories:
            raise ValueError("Topic categories must be specified")

        if not self.input_handler:
            raise ValueError("Input handler is required for pipeline execution")

        if not self.input_handler.validate():
            raise ValueError("Invalid input source")

        if self.mode == "sync":
            await self._run_sync_mode()
        else:
            await self._run_batch_mode()

    async def _run_sync_mode(self) -> None:
        """Enhanced sync mode with record-by-record processing, checkpointing, and progress tracking"""
        logger.info(f"Starting pipeline in sync mode (processing individual records)")

        if self.enable_checkpointing:
            logger.info(
                f"Checkpointing enabled - session: {self.checkpointer.session_id}"
            )
            logger.info(f"Checkpoint interval: {self.checkpoint_interval} records")

        # Try to resume from checkpoint if requested
        pending_items = []
        processed_items = 0
        total_results = 0

        if self.resume_from_checkpoint and self.checkpointer:
            checkpoint_items, checkpoint_state = self.checkpointer.load_checkpoint()
            if checkpoint_state:
                logger.info("Resuming from checkpoint...")
                processed_items = checkpoint_state.get("processed_count", 0)
                total_results = checkpoint_state.get("total_results", 0)

                if checkpoint_items:
                    pending_items = checkpoint_items
                    logger.info(f"Resuming with {len(pending_items)} pending items")

                # Load and output any intermediate results
                intermediate_results = self.checkpointer.get_intermediate_results()
                if intermediate_results:
                    logger.info(
                        f"Found {len(intermediate_results)} intermediate results"
                    )
                    # Convert back to CategoryResult objects and output them
                    recovered_results = [
                        CategoryResult(
                            id=r["id"],
                            categories=r["categories"],
                            confidence_scores=r.get("confidence_scores"),
                            model_response=r.get("model_response", {}),
                            processed_at=(
                                datetime.fromisoformat(r["processed_at"])
                                if r.get("processed_at")
                                else None
                            ),
                        )
                        for r in intermediate_results
                    ]
                    self.output_handler.write(recovered_results)
                    logger.info(
                        f"Recovered and output {len(recovered_results)} previous results"
                    )

        # Collect all items if not resuming or no pending items
        all_items = []
        if not pending_items:
            logger.info("Reading input data...")
            try:
                all_items = list(self.input_handler.read())
                logger.info(f"Loaded {len(all_items)} items from input")
            except Exception as e:
                logger.error(f"Failed to read input data: {e}")
                raise

        try:
            # Process items one by one (no formal batching)
            remaining_items = (
                pending_items if pending_items else all_items[processed_items:]
            )

            total_items = (
                len(all_items) if all_items else processed_items + len(pending_items)
            )
            logger.info(f"Total items to process: {total_items}")

            while remaining_items:
                current_item = remaining_items.pop(0)
                processed_items += 1

                # Show progress every 50 records or on checkpoint intervals
                if processed_items % 50 == 0 or (
                    self.checkpointer
                    and self.checkpointer.should_checkpoint(processed_items)
                ):
                    progress_msg = f"Processing record {processed_items}/{total_items}"
                    if self.checkpointer:
                        progress = self.checkpointer.estimate_progress(total_items)
                        if "progress_percent" in progress:
                            progress_msg += f" - {progress['progress_percent']:.1f}% complete, ETA: {progress.get('estimated_remaining_formatted', 'calculating...')}"
                    logger.info(progress_msg)

                # Process single item
                try:
                    results = await self.process_items([current_item])
                    if results:
                        # Save results to output immediately
                        self.output_handler.write(results)
                        total_results += len(results)

                        # Save intermediate results for recovery
                        if self.checkpointer:
                            self.checkpointer.save_intermediate_results(results)

                    # Save checkpoint if enabled and it's time
                    if self.checkpointer and self.checkpointer.should_checkpoint(
                        processed_items
                    ):
                        self.checkpointer.save_checkpoint(
                            pending_items=remaining_items,
                            processed_count=processed_items,
                            total_results=total_results,
                        )

                except Exception as e:
                    logger.error(f"Record {processed_items} failed: {e}")

                    # Save emergency checkpoint before retrying or failing
                    if self.checkpointer:
                        emergency_pending = [current_item] + remaining_items
                        self.checkpointer.save_checkpoint(
                            pending_items=emergency_pending,
                            processed_count=processed_items
                            - 1,  # Don't count failed item
                            total_results=total_results,
                            additional_state={
                                "last_error": str(e),
                                "emergency_save": True,
                            },
                        )
                        logger.info("Emergency checkpoint saved before failing")

                    raise

                    raise

            # Final checkpoint and cleanup
            if self.checkpointer:
                # Save final state
                self.checkpointer.save_checkpoint(
                    pending_items=[],
                    processed_count=processed_items,
                    batch_count=batch_count,
                    total_results=total_results,
                    additional_state={"completed": True},
                )

                # Show final progress
                final_progress = self.checkpointer.estimate_progress(total_items)
                logger.info(
                    f"Processing completed in {final_progress['elapsed_time_formatted']}"
                )

                # Cleanup checkpoint files
                self.checkpointer.cleanup()

            logger.info(
                f"Sync pipeline completed successfully: "
                f"{processed_items} items processed, {total_results} results saved"
            )

        except Exception as e:
            logger.error(f"Sync pipeline execution failed: {e}")
            logger.info(
                f"Partial progress: {processed_items} items were processed successfully"
            )
            if self.checkpointer:
                logger.info(
                    f"Recovery info: Use session_id '{self.checkpointer.session_id}' "
                    f"to resume from checkpoint"
                )
            raise

    async def _run_batch_mode(self) -> None:
        """Original batch mode processing"""
        logger.info(
            f"Starting pipeline in batch mode with batch size {self.batch_size}"
        )

        batch = []
        processed_items = 0
        total_results = 0

        try:
            for item in self.input_handler.read():
                batch.append(item)
                processed_items += 1

                if len(batch) >= self.batch_size:
                    logger.info(
                        f"Processing batch of {len(batch)} items (total processed: {processed_items})"
                    )
                    results = await self.process_items(batch)
                    if results:
                        self.output_handler.write(results)
                        total_results += len(results)
                        logger.info(
                            f"Saved {len(results)} results (total saved: {total_results})"
                        )
                    batch = []

            # Process remaining items
            if batch:
                logger.info(f"Processing final batch of {len(batch)} items")
                results = await self.process_items(batch)
                if results:
                    self.output_handler.write(results)
                    total_results += len(results)
                    logger.info(
                        f"Saved {len(results)} results (total saved: {total_results})"
                    )

            logger.info(
                f"Batch pipeline completed successfully: {processed_items} items processed, {total_results} results saved"
            )

        except Exception as e:
            logger.error(f"Batch pipeline execution failed: {e}")
            if batch:
                logger.info(
                    f"Partial progress: {processed_items - len(batch)} items were processed successfully"
                )
            raise

    @classmethod
    def list_checkpoint_sessions(cls, checkpoint_dir: str = "checkpoints") -> List[str]:
        """List all available checkpoint sessions"""
        checkpointer = SyncCheckpointer(checkpoint_dir=checkpoint_dir)
        return checkpointer.list_available_sessions()

    @classmethod
    def get_session_info(
        cls, session_id: str, checkpoint_dir: str = "checkpoints"
    ) -> Optional[dict]:
        """Get information about a specific checkpoint session"""
        checkpointer = SyncCheckpointer(checkpoint_dir=checkpoint_dir)
        return checkpointer.get_session_info(session_id)

    def get_checkpoint_status(self) -> Optional[dict]:
        """Get current checkpoint status"""
        if not self.checkpointer:
            return None

        return {
            "session_id": self.checkpointer.session_id,
            "checkpoint_interval": self.checkpointer.checkpoint_interval,
            "processed_count": self.checkpointer.processed_count,
            "batch_count": self.checkpointer.batch_count,
            "total_results": self.checkpointer.total_results,
            "enabled": self.enable_checkpointing,
        }
