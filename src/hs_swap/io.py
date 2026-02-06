import json
import os
from typing import Any, Dict, Iterator, List, Optional, Set


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL records."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all JSONL records into a list."""
    return list(iter_jsonl(path))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_written_custom_ids(output_jsonl: str) -> Set[str]:
    """Collect already-written custom_id values from an existing output JSONL."""
    written: Set[str] = set()
    if not os.path.exists(output_jsonl):
        return written

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id")
            if cid is not None:
                written.add(str(cid))
    return written


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append one record to JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
