"""
Performance monitoring utilities for the lung cancer detection system.
"""
import time
import psutil
import streamlit as st
from datetime import datetime
from typing import Dict, Any
from logger import logger

class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation."""
        self.start_time = time.time()
        self.metrics[operation_name] = {
            'start_time': self.start_time,
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'start_cpu': psutil.cpu_percent()
        }
        logger.info(f"Started monitoring: {operation_name}")
    
    def end_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """End monitoring and return metrics."""
        if operation_name not in self.metrics:
            logger.warning(f"No monitoring started for: {operation_name}")
            return {}
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()
        
        metrics = self.metrics[operation_name]
        
        result = {
            'operation': operation_name,
            'duration_ms': (end_time - metrics['start_time']) * 1000,
            'memory_used_mb': end_memory - metrics['start_memory'],
            'cpu_usage_percent': (end_cpu + metrics['start_cpu']) / 2,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance metrics for {operation_name}: {result}")
        return result
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get current system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
    
    @staticmethod
    def display_performance_metrics(metrics: Dict[str, Any]):
        """Display performance metrics in Streamlit."""
        if not metrics:
            return
        
        st.sidebar.markdown("### ⚡ Performance")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric(
                "Duration", 
                f"{metrics.get('duration_ms', 0):.1f} ms"
            )
        
        with col2:
            st.metric(
                "Memory", 
                f"{metrics.get('memory_used_mb', 0):.1f} MB"
            )

# Global performance monitor instance
performance_monitor = PerformanceMonitor()