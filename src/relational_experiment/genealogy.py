from __future__ import annotations

import random

from .model import Person, Problem

FEMALE_NAMES = [
    "Alice", "Beth", "Clara", "Diana", "Eva", "Fiona", "Grace", "Helen",
    "Iris", "Julia", "Karen", "Laura", "Maya", "Nora", "Olivia", "Paula",
]
MALE_NAMES = [
    "Adam", "Ben", "Carl", "David", "Evan", "Frank", "George", "Henry",
    "Ian", "James", "Kevin", "Luke", "Mark", "Noah", "Oscar", "Paul",
]


def parents_of(problem: Problem, person_id: str) -> set[str]:
    return {parent for child, parent in problem.parent_edges if child == person_id}


def children_of(problem: Problem, person_id: str) -> set[str]:
    return {child for child, parent in problem.parent_edges if parent == person_id}


def siblings_of(problem: Problem, person_id: str) -> set[str]:
    result: set[str] = set()
    for parent in parents_of(problem, person_id):
        result |= children_of(problem, parent)
    result.discard(person_id)
    return result


def grandparents_of(problem: Problem, person_id: str) -> set[str]:
    return {
        grandparent
        for parent in parents_of(problem, person_id)
        for grandparent in parents_of(problem, parent)
    }


def relation_of(problem: Problem, source: str, target: str) -> str | None:
    gender = {person.id: person.gender for person in problem.people}
    if target in parents_of(problem, source):
        return "mother" if gender[target] == "female" else "father"
    if target in siblings_of(problem, source):
        return "sister" if gender[target] == "female" else "brother"
    if target in grandparents_of(problem, source):
        return "grandmother" if gender[target] == "female" else "grandfather"
    parent_siblings = {
        sibling
        for parent in parents_of(problem, source)
        for sibling in siblings_of(problem, parent)
    }
    if target in parent_siblings:
        return "aunt" if gender[target] == "female" else "uncle"
    grandparent_siblings = {
        sibling
        for grandparent in grandparents_of(problem, source)
        for sibling in siblings_of(problem, grandparent)
    }
    if target in grandparent_siblings:
        return "great-aunt" if gender[target] == "female" else "great-uncle"
    return None


RELATION_FAMILIES = ("parent", "sibling", "grandparent", "avuncular", "great_avuncular")


def generate_problem(index: int, seed: int, distractor_count: int = 2) -> Problem:
    """Generate a valid problem from one of five relationship topologies."""
    rng = random.Random(seed)
    # Keep enough names of either gender for every possible random assignment.
    female_names = rng.sample(FEMALE_NAMES, 8)
    male_names = rng.sample(MALE_NAMES, 8)
    family = RELATION_FAMILIES[index % len(RELATION_FAMILIES)]
    source_gender = rng.choice(["female", "male"])
    source_name = female_names.pop() if source_gender == "female" else male_names.pop()
    target_gender = rng.choice(["female", "male"])
    target_name = female_names.pop() if target_gender == "female" else male_names.pop()
    people = [Person("p0", source_name, source_gender), Person("p3", target_name, target_gender)]

    def add_person(pid: str) -> None:
        gender = rng.choice(["female", "male"])
        name = female_names.pop() if gender == "female" else male_names.pop()
        people.append(Person(pid, name, gender))

    if family == "parent":
        edges = [("p0", "p3")]
        expected = "mother" if target_gender == "female" else "father"
    elif family == "sibling":
        add_person("p4")
        edges = [("p0", "p4"), ("p3", "p4")]
        expected = "sister" if target_gender == "female" else "brother"
    elif family == "grandparent":
        add_person("p1")
        edges = [("p0", "p1"), ("p1", "p3")]
        expected = "grandmother" if target_gender == "female" else "grandfather"
    elif family == "avuncular":
        add_person("p1")
        add_person("p4")
        edges = [("p0", "p1"), ("p1", "p4"), ("p3", "p4")]
        expected = "aunt" if target_gender == "female" else "uncle"
    else:
        add_person("p1")
        add_person("p2")
        add_person("p4")
        edges = [("p0", "p1"), ("p1", "p2"), ("p2", "p4"), ("p3", "p4")]
        expected = "great-aunt" if target_gender == "female" else "great-uncle"

    distractors: list[tuple[str, str]] = []
    for offset in range(distractor_count):
        gender = "female" if offset % 2 == 0 else "male"
        child_id, parent_id = f"d{offset}c", f"d{offset}p"
        child_name = female_names.pop() if gender == "female" else male_names.pop()
        parent_gender = "male" if gender == "female" else "female"
        parent_name = male_names.pop() if parent_gender == "male" else female_names.pop()
        people.extend([Person(child_id, child_name, gender), Person(parent_id, parent_name, parent_gender)])
        distractors.append((child_id, parent_id))
    problem = Problem(
        problem_id=f"p{index:05d}",
        seed=seed,
        people=tuple(people),
        parent_edges=tuple(edges + distractors),
        source="p0",
        target="p3",
        correct_relation=expected,
        distractor_facts=tuple(distractors),
    )
    actual = relation_of(problem, problem.source, problem.target)
    if actual != expected:
        raise AssertionError(f"generator error: expected {expected}, got {actual}")
    return problem


def generate_problems(count: int, seed: int, distractor_count: int = 2) -> list[Problem]:
    rng = random.Random(seed)
    return [
        generate_problem(i, rng.randrange(1, 2**31), distractor_count)
        for i in range(count)
    ]
