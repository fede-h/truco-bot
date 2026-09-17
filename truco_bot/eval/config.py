"""Configuration schemas for showdown evaluation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ShowdownConfig:
    candidate: str
    opponents: list[str]
    n_matches: int = 100
    seed: int = 42
    track_bluff: bool = True

    def __post_init__(self) -> None:
        if self.candidate is None:
            raise TypeError("candidate cannot be None")
        if not isinstance(self.candidate, str):
            raise TypeError(f"candidate must be a string, got {type(self.candidate).__name__}")
        if not self.candidate.strip():
            raise ValueError("candidate cannot be empty")

        if self.opponents is None:
            raise TypeError("opponents cannot be None")
        if not isinstance(self.opponents, list):
            raise TypeError(f"opponents must be a list, got {type(self.opponents).__name__}")
        if len(self.opponents) == 0:
            raise ValueError("opponents list cannot be empty")
        for opp in self.opponents:
            if opp is None:
                raise TypeError("opponents elements cannot be None")
            if not isinstance(opp, str):
                raise TypeError(f"opponents elements must be strings, got {type(opp).__name__}")
            if not opp.strip():
                raise ValueError("opponents elements cannot be empty strings")

        if isinstance(self.n_matches, bool) or not isinstance(self.n_matches, int):
            raise TypeError(f"n_matches must be an int, got {type(self.n_matches).__name__}")
        if self.n_matches <= 0:
            raise ValueError(f"n_matches must be greater than 0, got {self.n_matches}")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")

        if not isinstance(self.track_bluff, bool):
            raise TypeError(f"track_bluff must be a bool, got {type(self.track_bluff).__name__}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShowdownConfig":
        # ponytail: strict dict validation rejecting unknown keys
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, got {type(d).__name__}")
        known_keys = {"candidate", "opponents", "n_matches", "seed", "track_bluff"}
        unknown = set(d.keys()) - known_keys
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        if "candidate" not in d:
            raise ValueError("Missing required key: 'candidate'")
        if "opponents" not in d:
            raise ValueError("Missing required key: 'opponents'")
        return cls(**d)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "ShowdownConfig":
        # ponytail: read from path or direct yaml string
        if isinstance(source, Path):
            content = source.read_text(encoding="utf-8")
        elif isinstance(source, str):
            p = Path(source)
            if "\n" not in source and p.is_file():
                content = p.read_text(encoding="utf-8")
            else:
                content = source
        else:
            raise TypeError(f"source must be str or Path, got {type(source).__name__}")

        if yaml is not None:
            data = yaml.safe_load(content)
        else:
            # ponytail: fallback parser for minimal yaml key: value
            data = {}
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if v.isdigit():
                        data[k] = int(v)
                    elif v.lower() in ("true", "false"):
                        data[k] = v.lower() == "true"
                    elif v.startswith("[") and v.endswith("]"):
                        data[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                    else:
                        data[k] = v

        if not isinstance(data, dict):
            raise TypeError("YAML source must parse into a dictionary")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
