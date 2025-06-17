from datetime import datetime
from typing import Dict, Any, Optional
from .base_collector import BaseMetricsCollector

class HuggingFaceMetricsCollector(BaseMetricsCollector):
    def __init__(self, model: str, is_local: bool = False, output_dir: str = "metrics_output"):
        super().__init__(output_dir)
        self.model = model
        self.is_local = is_local
        
        # Add local-specific metrics if using local model
        if is_local:
            self.metrics.update({
                "gpu_memory_usage": [],
                "cpu_usage": [],
                "model_load_time": None,
                "power_consumption": []  # Optional
            })
    
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
    
    def record_batch(self, batch_id: str, num_samples: int, start_time: datetime, end_time: datetime,
                    cost: Optional[float] = None, error: Optional[str] = None):
        """Record metrics for a single batch"""
        batch_metrics = {
            "batch_id": batch_id,
            "num_samples": num_samples,
            "start_time": start_time,
            "end_time": end_time,
            "runtime": (end_time - start_time).total_seconds(),
            "cost": cost or 0.0,
            "error": error
        }
        
        self.metrics["batch_metrics"].append(batch_metrics)
        self.metrics["total_samples"] += num_samples
        self.metrics["total_cost"] += cost or 0.0
        
        if error:
            self.metrics["errors"].append({
                "batch_id": batch_id,
                "error": error,
                "timestamp": datetime.now()
            })
        
        # Update cost per sample
        if self.metrics["total_samples"] > 0:
            self.metrics["cost_per_sample"] = (
                self.metrics["total_cost"] / self.metrics["total_samples"]
            )
    
    def record_resource_usage(self, gpu_memory: Optional[float] = None, 
                            cpu_usage: Optional[float] = None,
                            power_consumption: Optional[float] = None):
        """Record resource usage metrics for local models"""
        if not self.is_local:
            return
        
        timestamp = datetime.now()
        
        if gpu_memory is not None:
            self.metrics["gpu_memory_usage"].append({
                "timestamp": timestamp,
                "memory_mb": gpu_memory
            })
        
        if cpu_usage is not None:
            self.metrics["cpu_usage"].append({
                "timestamp": timestamp,
                "usage_percent": cpu_usage
            })
        
        if power_consumption is not None:
            self.metrics["power_consumption"].append({
                "timestamp": timestamp,
                "watts": power_consumption
            })
    
    def record_model_load_time(self, load_time: float):
        """Record the time taken to load the model"""
        if self.is_local:
            self.metrics["model_load_time"] = load_time 