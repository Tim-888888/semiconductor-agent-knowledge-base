"""Idempotently replace the initial Milvus root password without logging secrets."""

from __future__ import annotations

import os
import time

from pymilvus import MilvusClient


def _connect(uri: str, password: str) -> MilvusClient:
    return MilvusClient(uri=uri, token=f"root:{password}", timeout=5)


def initialize_root_password(uri: str, desired_password: str) -> str:
    if len(desired_password) < 16:
        raise RuntimeError("MILVUS_ROOT_PASSWORD must contain at least 16 characters.")

    for attempt in range(30):
        try:
            client = _connect(uri, desired_password)
            try:
                client.list_users()
                return "already_initialized"
            finally:
                client.close()
        except Exception:
            try:
                client = _connect(uri, "Milvus")
                try:
                    client.update_password("root", "Milvus", desired_password)
                    return "initialized"
                finally:
                    client.close()
            except Exception as exc:
                if attempt == 29:
                    raise RuntimeError(
                        "Milvus authentication initialization failed; inspect service health."
                    ) from exc
                time.sleep(2)
    raise RuntimeError("Milvus authentication initialization failed.")


def main() -> None:
    uri = os.environ.get("MILVUS_URI", "http://milvus:19530")
    password = os.environ.get("MILVUS_ROOT_PASSWORD", "")
    status = initialize_root_password(uri, password)
    print(f"Milvus root authentication status: {status}")


if __name__ == "__main__":
    main()
