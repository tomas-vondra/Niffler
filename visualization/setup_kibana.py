#!/usr/bin/env python3
"""
Setup Kibana Data Views for Niffler

Automatically creates Kibana data views (index patterns) for Niffler indices.

Usage:
    python visualization/setup_kibana.py
"""

import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests package not installed.")
    print("Install it with: uv sync")
    sys.exit(1)


def niffler_data_views() -> list:
    """Return one data view per index the Elasticsearch exporter writes.

    Read from the exporter rather than listed here: a hand-kept copy had already
    lost the positions index and ignored ELASTICSEARCH_INDEX_PREFIX.

    Returns:
        Dicts with ``title``, ``pattern`` and ``time_field``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter

    return [
        {"title": f"Niffler {index['title']}", "pattern": index["name"],
         "time_field": index["time_field"]}
        for index in ElasticsearchExporter().index_catalog()
    ]


class KibanaSetup:
    """Setup Kibana data views for Niffler."""

    def __init__(self, host: str = "localhost", port: int = 5601):
        self.host = host
        self.port = port
        self.kibana_url = f"http://{host}:{port}"
        self.api_url = f"{self.kibana_url}/api/data_views/data_view"
        self.headers = {"kbn-xsrf": "true", "Content-Type": "application/json"}

    def check_health(self) -> bool:
        """Check if Kibana is available."""
        try:
            response = requests.get(f"{self.kibana_url}/api/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                if status.get("status", {}).get("overall", {}).get("level") == "available":
                    print(f"[OK] Connected to Kibana at {self.kibana_url}")
                    return True
            print(f"[ERROR] Kibana is not ready")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to Kibana at {self.kibana_url}")
            print("[ERROR] Start Kibana: docker-compose --profile debug up -d kibana")
            return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def create_data_view(self, title: str, time_field: str, pattern: str) -> bool:
        """Create a Kibana data view."""
        payload = {
            "data_view": {
                "title": pattern,
                "name": title,
                "timeFieldName": time_field
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)

            if response.status_code in [200, 201]:
                print(f"  [OK] Created: {title}")
                return True
            elif response.status_code == 409:
                print(f"  [SKIP] Already exists: {title}")
                return True
            elif response.status_code == 400:
                # Check if it's a duplicate error
                error_msg = response.json().get("message", "")
                if "duplicate" in error_msg.lower():
                    print(f"  [SKIP] Already exists: {title}")
                    return True
                print(f"  [ERROR] Failed: {title} - {error_msg}")
                return False
            else:
                print(f"  [ERROR] Failed: {title}")
                return False
        except Exception as e:
            print(f"  [ERROR] {title}: {e}")
            return False

    def setup(self) -> bool:
        """Create all Niffler data views."""
        data_views = niffler_data_views()

        print("=" * 80)
        print("Kibana Data Views Setup")
        print("=" * 80)

        if not self.check_health():
            return False

        print("\nCreating data views...")
        success_count = 0
        for dv in data_views:
            if self.create_data_view(dv["title"], dv["time_field"], dv["pattern"]):
                success_count += 1

        print("=" * 80)
        print(f"[SUCCESS] Created {success_count}/{len(data_views)} data views")
        print("=" * 80)

        if success_count == len(data_views):
            print(f"\nAccess Kibana: {self.kibana_url}")
            print("Go to: Analytics -> Discover")
            print("Select a data view and adjust time range (Last 90 days)")
            return True
        else:
            print("\nSome data views could not be created.")
            print("Run a backtest first to create the indices.")
            return False


def main():
    """Main entry point."""
    try:
        setup = KibanaSetup()
        success = setup.setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[ERROR] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
