"""Configuration schemas for showdown evaluation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ShowdownConfig:
    candidate: str
    opponents: list[str]
    n_matches: int = 100
    seed: int = 42
    track_bluff: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, str) or not self.candidate.strip():
            raise ValueError("candidate cannot be empty")
        if not isinstance(self.opponents, list) or not self.opponents:
            raise ValueError("opponents list cannot be empty")
        if any(not isinstance(o, str) or not o.strip() for o in self.opponents):
            raise ValueError("opponents elements cannot be empty strings")
        if isinstance(self.n_matches, bool) or not isinstance(self.n_matches, int) or self.n_matches <= 0:
            raise ValueError(f"n_matches must be greater than 0, got {self.n_matches}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")
        if not isinstance(self.track_bluff, bool):
            raise TypeError(f"track_bluff must be a bool, got {type(self.track_bluff).__name__}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShowdownConfig":
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, got {type(d).__name__}")
        known = {"candidate", "opponents", "n_matches", "seed", "track_bluff"}
        if unknown := set(d) - known:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        if "candidate" not in d:
            raise ValueError("Missing required key: 'candidate'")
        if "opponents" not in d:
            raise ValueError("Missing required key: 'opponents'")
        return cls(**d)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "ShowdownConfig":
        p = (
            Path(source)
            if isinstance(source, Path) or ("\n" not in str(source) and Path(str(source)).is_file())
            else None
        )
        content = p.read_text(encoding="utf-8") if p else str(source)
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise TypeError("YAML source must parse into a dictionary")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
