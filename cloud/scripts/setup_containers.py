#!/usr/bin/env python3
"""One-time Azure Blob container creation.

Usage:
    python -m cloud.scripts.setup_containers
"""

import logging

from cloud.storage.container_setup import setup_containers


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Setting up Azure Blob Storage containers...")
    setup_containers()
    print("Done.")


if __name__ == "__main__":
    main()
