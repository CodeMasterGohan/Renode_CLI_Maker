"""
In-memory caching system for the CLI application.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Represents a single cache entry with TTL support."""
    data: Any
    timestamp: float
    ttl: float
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


class InMemoryCache:
    """In-memory cache with TTL and size limit support."""
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        
    def _cleanup_expired(self):
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _enforce_size_limit(self):
        """Enforce the cache size limit by removing oldest entries."""
        if len(self.cache) > self.max_size:
            # Sort by timestamp (oldest first) and remove excess entries
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1].timestamp
            )
            remove_count = len(self.cache) - int(self.max_size * 0.8)
            for key, _ in sorted_items[:remove_count]:
                del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        self._cleanup_expired()
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache."""
        if ttl is None:
            ttl = self.default_ttl
            
        self.cache[key] = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self._enforce_size_limit()
    
    def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache entries."""
        if pattern is None:
            self.cache.clear()
        else:
            removed_keys = [key for key in self.cache.keys() if pattern in key]
            for key in removed_keys:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "oldest_entry": min(
                (entry.timestamp for entry in self.cache.values()), 
                default=time.time()
            ),
            "newest_entry": max(
                (entry.timestamp for entry in self.cache.values()), 
                default=time.time()
            )
        } 