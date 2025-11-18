#!/usr/bin/env python3
"""
Clean Elasticsearch Indices for Niffler

Removes all Niffler indices from Elasticsearch for fresh starts.

Usage:
    python visualization/clean_elasticsearch.py [--force] [--dry-run]
"""

import argparse
import sys
from typing import List, Dict, Any

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import NotFoundError, ConnectionError
except ImportError:
    print("Error: elasticsearch package not installed.")
    print("Install it with: uv sync")
    sys.exit(1)


class ElasticsearchCleaner:
    """Clean Elasticsearch indices for Niffler."""

    def __init__(self, host: str = "localhost", port: int = 9200, index_prefix: str = "niffler"):
        self.host = host
        self.port = port
        self.index_prefix = index_prefix
        self.es_url = f"http://{host}:{port}"
        self.es_client = None

    def connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        try:
            self.es_client = Elasticsearch([self.es_url])
            if self.es_client.ping():
                print(f"[OK] Connected to Elasticsearch at {self.es_url}")
                return True
            else:
                print(f"[ERROR] Cannot connect to Elasticsearch at {self.es_url}")
                return False
        except ConnectionError as e:
            print(f"[ERROR] Connection error: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return False

    def get_niffler_indices(self) -> List[str]:
        """Get all Niffler indices from Elasticsearch."""
        try:
            pattern = f"{self.index_prefix}-*"
            indices = self.es_client.cat.indices(index=pattern, format="json")

            if not indices:
                return []

            return [idx['index'] for idx in indices]

        except NotFoundError:
            return []
        except Exception as e:
            print(f"[ERROR] Error getting indices: {e}")
            return []

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index."""
        try:
            stats = self.es_client.count(index=index_name)
            size_info = self.es_client.cat.indices(index=index_name, format="json", bytes="b")

            doc_count = stats.get('count', 0)
            size_bytes = int(size_info[0].get('store.size', '0')) if size_info else 0

            return {
                'name': index_name,
                'doc_count': doc_count,
                'size_bytes': size_bytes,
                'size_human': self._format_bytes(size_bytes)
            }
        except Exception:
            return {
                'name': index_name,
                'doc_count': 0,
                'size_bytes': 0,
                'size_human': '0 B'
            }

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    def display_indices(self, indices: List[str]) -> None:
        """Display indices with their statistics."""
        print("\nFound Niffler indices:")
        print("-" * 80)
        print(f"{'Index Name':<40} {'Documents':>15} {'Size':>15}")
        print("-" * 80)

        total_docs = 0
        total_size = 0

        for index in sorted(indices):
            stats = self.get_index_stats(index)
            print(f"{stats['name']:<40} {stats['doc_count']:>15,} {stats['size_human']:>15}")
            total_docs += stats['doc_count']
            total_size += stats['size_bytes']

        print("-" * 80)
        print(f"{'TOTAL':<40} {total_docs:>15,} {self._format_bytes(total_size):>15}")
        print("-" * 80)

    def delete_indices(self, indices: List[str], force: bool = False) -> bool:
        """Delete indices from Elasticsearch."""
        if not indices:
            print("\nNo indices to delete.")
            return True

        self.display_indices(indices)

        if not force:
            print(f"\n[WARNING] This will permanently delete {len(indices)} indices!")
            print("[WARNING] All data will be lost and cannot be recovered!")
            response = input("\nType 'yes' to confirm deletion: ").strip().lower()

            if response != 'yes':
                print("\n[CANCELLED] Deletion cancelled.")
                return False

        print(f"\nDeleting {len(indices)} indices...")
        success_count = 0
        error_count = 0

        for index in indices:
            try:
                self.es_client.indices.delete(index=index)
                print(f"  [OK] Deleted: {index}")
                success_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to delete {index}: {e}")
                error_count += 1

        print("\n" + "=" * 80)
        print(f"[SUCCESS] Successfully deleted: {success_count} indices")
        if error_count > 0:
            print(f"[FAILED] Failed to delete: {error_count} indices")
        print("=" * 80)

        return error_count == 0

    def clean(self, force: bool = False, dry_run: bool = False) -> bool:
        """Main cleanup method."""
        print("=" * 80)
        print("Niffler Elasticsearch Cleaner")
        print("=" * 80)
        print(f"Elasticsearch: {self.es_url}")
        print(f"Index prefix: {self.index_prefix}")
        print("=" * 80)

        if not self.connect():
            return False

        indices = self.get_niffler_indices()

        if not indices:
            print(f"\n[OK] No indices found matching '{self.index_prefix}-*'")
            print("[OK] Elasticsearch is already clean!")
            return True

        if dry_run:
            print("\nDRY RUN MODE - No changes will be made")
            self.display_indices(indices)
            print(f"\nWould delete {len(indices)} indices.")
            return True

        return self.delete_indices(indices, force)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean Elasticsearch indices for Niffler")
    parser.add_argument("--force", "-f", action="store_true", help="Delete without confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()

    try:
        cleaner = ElasticsearchCleaner()
        success = cleaner.clean(force=args.force, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[ERROR] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
