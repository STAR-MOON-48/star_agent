from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re

from ...protocols import JsonDict, ensure_json_dict_list, ensure_string_list, utc_now


@dataclass
class MemoryRecord:
    memory_id: str
    agent_id: str
    kind: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    source_refs: list[JsonDict] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.tags = ensure_string_list(self.tags)
        self.source_refs = ensure_json_dict_list(self.source_refs)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "MemoryRecord":
        return MemoryRecord(**data)

    def to_markdown(self) -> str:
        metadata = {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "kind": self.kind,
            "tags": self.tags,
            "source_refs": self.source_refs,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return (
            "<!-- agent-memory "
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            + " -->\n\n"
            + self.content.strip()
            + "\n"
        )


class MemoryStore:
    """Markdown source-of-truth store with a compact JSON search index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "memories"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = utc_now()
        agent_root = self._agent_root(record.agent_id)
        kind_name = self._safe_component(record.kind)
        kind_root = agent_root / kind_name
        kind_root.mkdir(parents=True, exist_ok=True)
        relative_path = Path(kind_name) / f"{self._safe_component(record.memory_id)}.md"
        path = agent_root / relative_path
        temporary = path.with_suffix(".tmp")
        temporary.write_text(record.to_markdown(), encoding="utf-8")
        os.replace(temporary, path)

        index = self._load_index(record.agent_id)
        index[record.memory_id] = {
            **{key: value for key, value in record.to_dict().items() if key != "content"},
            "path": str(relative_path),
            "preview": record.content.strip()[:500],
        }
        self._save_index(record.agent_id, index)
        return record

    def get(self, agent_id: str, memory_id: str) -> MemoryRecord | None:
        item = self._load_index(agent_id).get(memory_id)
        if not isinstance(item, dict):
            return None
        path = self._agent_root(agent_id) / str(item.get("path") or "")
        if not path.exists():
            return None
        markdown = path.read_text(encoding="utf-8")
        content = re.sub(
            r"^<!-- agent-memory .*?-->\s*",
            "",
            markdown,
            count=1,
        ).strip()
        return MemoryRecord.from_dict(
            {
                "memory_id": item["memory_id"],
                "agent_id": item["agent_id"],
                "kind": item["kind"],
                "title": item["title"],
                "content": content,
                "tags": list(item.get("tags") or []),
                "source_refs": list(item.get("source_refs") or []),
                "confidence": float(item.get("confidence", 1.0)),
                "created_at": item.get("created_at") or utc_now(),
                "updated_at": item.get("updated_at") or utc_now(),
            }
        )

    def search(
        self,
        agent_id: str,
        *,
        query: str = "",
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        query_terms = self._terms(query)
        required_tags = {tag.casefold() for tag in (tags or []) if tag}
        ranked: list[tuple[float, str, str]] = []
        for memory_id, item in self._load_index(agent_id).items():
            if not isinstance(item, dict):
                continue
            if kind and item.get("kind") != kind:
                continue
            item_tags = {str(tag).casefold() for tag in item.get("tags") or []}
            if required_tags and not required_tags.issubset(item_tags):
                continue
            searchable = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("preview") or ""),
                    " ".join(item_tags),
                ]
            ).casefold()
            score = self._score(searchable, query_terms)
            if query_terms and score <= 0:
                continue
            ranked.append((score, str(item.get("updated_at") or ""), memory_id))
        ranked.sort(reverse=True)
        records: list[MemoryRecord] = []
        for _, _, memory_id in ranked[: max(1, limit)]:
            record = self.get(agent_id, memory_id)
            if record is not None:
                records.append(record)
        return records

    def _agent_root(self, agent_id: str) -> Path:
        root = self.root / self._safe_component(agent_id)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _index_path(self, agent_id: str) -> Path:
        return self._agent_root(agent_id) / "index.json"

    def _load_index(self, agent_id: str) -> dict[str, JsonDict]:
        path = self._index_path(agent_id)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}

    def _save_index(self, agent_id: str, index: dict[str, JsonDict]) -> None:
        path = self._index_path(agent_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _safe_component(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "memory"

    def _terms(self, query: str) -> list[str]:
        normalized = query.casefold().strip()
        if not normalized:
            return []
        words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,8}", normalized)
        return list(dict.fromkeys(words))

    def _score(self, searchable: str, terms: list[str]) -> float:
        if not terms:
            return 1.0
        matched = sum(searchable.count(term) for term in terms)
        coverage = sum(1 for term in terms if term in searchable) / len(terms)
        return matched + coverage * 5.0
