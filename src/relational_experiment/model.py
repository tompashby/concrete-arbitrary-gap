from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Gender = Literal["female", "male"]


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    gender: Gender


@dataclass(frozen=True)
class Problem:
    problem_id: str
    seed: int
    people: tuple[Person, ...]
    parent_edges: tuple[tuple[str, str], ...]  # (child, parent)
    source: str
    target: str
    correct_relation: str
    distractor_facts: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Problem":
        return cls(
            problem_id=value["problem_id"],
            seed=value["seed"],
            people=tuple(Person(**p) for p in value["people"]),
            parent_edges=tuple(tuple(x) for x in value["parent_edges"]),
            source=value["source"],
            target=value["target"],
            correct_relation=value["correct_relation"],
            distractor_facts=tuple(tuple(x) for x in value.get("distractor_facts", [])),
        )


@dataclass(frozen=True)
class Trial:
    trial_id: str
    problem_id: str
    condition: str
    prompt: str
    choices: dict[str, str]
    correct_letter: str
    correct_semantic_answer: str
    path_length: int
    distractor_count: int
    prompt_chars: int
    accepted_answers: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Trial":
        value = dict(value)
        value["accepted_answers"] = tuple(value.get("accepted_answers", ()))
        return cls(**value)
