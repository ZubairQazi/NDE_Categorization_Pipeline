import time
import psutil
import threading
from typing import Optional, Callable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        """Initialize the resource monitor.
        
        Args:
            interval: Time between measurements in seconds
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        
        # Initialize GPU monitoring if available
        self._has_gpu = False
        try:
            import torch
            self._has_gpu = torch.cuda.is_available()
            if self._has_gpu:
                self._gpu_device = torch.cuda.current_device()
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
    
    def start(self, callback: Callable):
        """Start monitoring resources.
        
        Args:
            callback: Function to call with resource metrics
        """
        if self._monitor_thread is not None:
            logger.warning("Monitor already running")
            return
        
        self._callback = callback
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring resources"""
        if self._monitor_thread is None:
            return
        
        self._stop_event.set()
        self._monitor_thread.join()
        self._monitor_thread = None
        self._callback = None
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                if self._callback:
                    self._callback(**metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.interval)
    
    def _collect_metrics(self) -> dict:
        """Collect current resource metrics"""
        metrics = {
            "timestamp": datetime.now(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        }
        
        # Add GPU metrics if available
        if self._has_gpu:
            try:
                import torch
                metrics["gpu_memory"] = torch.cuda.memory_allocated(self._gpu_device) / (1024 * 1024)  # MB
                metrics["gpu_utilization"] = torch.cuda.utilization(self._gpu_device)
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    @staticmethod
    def get_power_consumption() -> Optional[float]:
        """Get current power consumption in watts (if available)"""
        try:
            # This is platform-specific and may not work on all systems
            if psutil.LINUX:
                with open("/sys/class/power_supply/BAT0/power_now", "r") as f:
                    return float(f.read().strip()) / 1000000  # Convert to watts
            elif psutil.MACOS:
                # On macOS, we can use powermetrics, but it requires sudo
                # This is just a placeholder - actual implementation would need sudo access
                return None
            else:
                return None
        except Exception:
            return None 