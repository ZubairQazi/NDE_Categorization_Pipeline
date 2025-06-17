import asyncio
from datetime import datetime
import logging
from pathlib import Path
import json
import uuid

from pipeline import Pipeline, OpenAIProvider, HuggingFaceProvider
from . import OpenAIMetricsCollector, HuggingFaceMetricsCollector, ResourceMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_with_metrics(pipeline: Pipeline, metrics_collector, resource_monitor=None):
    """Run the pipeline with metrics collection"""
    try:
        # Start metrics collection
        metrics_collector.start_collection()
        if resource_monitor:
            resource_monitor.start(metrics_collector.record_resource_usage)
        
        # Run the pipeline
        results = await pipeline.run()
        
        # Stop metrics collection
        metrics_collector.stop_collection()
        if resource_monitor:
            resource_monitor.stop()
        
        # Save metrics
        metrics_file = metrics_collector.save_metrics()
        logger.info(f"Metrics saved to {metrics_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        metrics_collector.stop_collection()
        if resource_monitor:
            resource_monitor.stop()
        raise

async def example_sync_processing():
    """Example of synchronous processing with metrics"""
    # Initialize collector for gpt-4.1-nano
    collector = OpenAIMetricsCollector(model="gpt-4.1-nano")
    
    # Example of recording individual requests
    start_time = datetime.now()
    
    # Simulate a few requests
    for i in range(3):
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        prompt = f"Example prompt {i}"
        completion = f"Example completion {i}"
        
        # Simulate request processing time
        await asyncio.sleep(0.1)
        end_time = datetime.now()
        
        # Record the request
        collector.record_request(
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            prompt=prompt,
            completion=completion
        )
    
    # Save metrics
    metrics_file = collector.save_metrics("sync_processing_metrics.json")
    logger.info(f"Sync processing metrics saved to {metrics_file}")

async def example_batch_processing():
    """Example of batch processing with metrics"""
    # Initialize collector for gpt-4o
    collector = OpenAIMetricsCollector(model="gpt-4o")
    
    # Example of recording batch requests
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    start_time = datetime.now()
    
    # Simulate a batch of requests
    prompts = [f"Batch prompt {i}" for i in range(5)]
    completions = [f"Batch completion {i}" for i in range(5)]
    
    # Simulate batch processing time
    await asyncio.sleep(0.5)
    end_time = datetime.now()
    
    # Record the batch
    collector.record_batch(
        batch_id=batch_id,
        num_samples=len(prompts),
        start_time=start_time,
        end_time=end_time,
        prompts=prompts,
        completions=completions
    )
    
    # Save metrics
    metrics_file = collector.save_metrics("batch_processing_metrics.json")
    logger.info(f"Batch processing metrics saved to {metrics_file}")

async def example_mixed_processing():
    """Example of mixed batch and sync processing with metrics"""
    # Initialize collector for gpt-4.1-mini
    collector = OpenAIMetricsCollector(model="gpt-4.1-mini")
    
    # Record some individual requests
    for i in range(2):
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        prompt = f"Mixed prompt {i}"
        completion = f"Mixed completion {i}"
        
        await asyncio.sleep(0.1)
        end_time = datetime.now()
        
        collector.record_request(
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            prompt=prompt,
            completion=completion
        )
    
    # Record a batch
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    start_time = datetime.now()
    
    prompts = [f"Mixed batch prompt {i}" for i in range(3)]
    completions = [f"Mixed batch completion {i}" for i in range(3)]
    
    await asyncio.sleep(0.3)
    end_time = datetime.now()
    
    collector.record_batch(
        batch_id=batch_id,
        num_samples=len(prompts),
        start_time=start_time,
        end_time=end_time,
        prompts=prompts,
        completions=completions
    )
    
    # Save metrics
    metrics_file = collector.save_metrics("mixed_processing_metrics.json")
    logger.info(f"Mixed processing metrics saved to {metrics_file}")

async def main():
    """Run all examples"""
    logger.info("Running synchronous processing example...")
    await example_sync_processing()
    
    logger.info("\nRunning batch processing example...")
    await example_batch_processing()
    
    logger.info("\nRunning mixed processing example...")
    await example_mixed_processing()

if __name__ == "__main__":
    asyncio.run(main()) 