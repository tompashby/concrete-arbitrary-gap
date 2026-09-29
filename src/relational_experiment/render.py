from __future__ import annotations

import random

from .model import Problem, Trial

LETTERS = "ABCD"
RELATION_DISTRACTORS = {
    "mother": ["sister", "grandmother", "aunt"],
    "father": ["brother", "grandfather", "uncle"],
    "sister": ["mother", "aunt", "grandmother"],
    "brother": ["father", "uncle", "grandfather"],
    "grandmother": ["mother", "aunt", "great-aunt"],
    "grandfather": ["father", "uncle", "great-uncle"],
    "aunt": ["sister", "grandmother", "great-aunt"],
    "uncle": ["brother", "grandfather", "great-uncle"],
    "great-aunt": ["aunt", "grandmother", "sister"],
    "great-uncle": ["uncle", "grandfather", "brother"],
}
NOVEL = ["miv", "tor", "lek", "gaf", "zun", "pev", "rol", "dak"]
PATH_LENGTH = {
    "mother": 1, "father": 1, "sister": 1, "brother": 1,
    "grandmother": 2, "grandfather": 2, "aunt": 2, "uncle": 2,
    "great-aunt": 3, "great-uncle": 3,
}


def _relation_definition(relation: str) -> str:
    gender = "female" if relation in {"mother", "sister", "grandmother", "aunt", "great-aunt"} else "male"
    if relation in {"mother", "father"}:
        return f"a {relation} is a person's {gender} parent"
    if relation in {"sister", "brother"}:
        return f"a {relation} is a {gender} sibling"
    if relation in {"grandmother", "grandfather"}:
        return f"a {relation} is a {gender} parent of one of a person's parents"
    if relation in {"aunt", "uncle"}:
        return f"an {relation} is a {gender} sibling of one of a person's parents"
    return f"a {relation} is a {gender} sibling of one of a person's grandparents"


def _abstract_definition(relation: str, target_label: str, sibling_label: str) -> str:
    gender = "female" if relation in {"mother", "sister", "grandmother", "aunt", "great-aunt"} else "male"
    if relation in {"mother", "father"}:
        parent = "female" if relation == "mother" else "male"
        return f"{target_label}(x,y) holds when y is the {parent} parent of x"
    if relation in {"sister", "brother"}:
        return f"{target_label}(x,y) holds when x and y share a parent and y is {gender}"
    if relation in {"grandmother", "grandfather"}:
        return f"{target_label}(x,y) holds when y is a {gender} parent of one of x's parents"
    if relation in {"aunt", "uncle"}:
        return f"{target_label}(x,y) holds when y is a {gender} sibling of one of x's parents"
    return f"{target_label}(x,y) holds when y is a {gender} sibling of one of x's grandparents"


def _person_map(problem: Problem) -> dict[str, object]:
    return {person.id: person for person in problem.people}


def _fact_sentences(problem: Problem) -> list[str]:
    people = _person_map(problem)
    lines = []
    for child, parent in problem.parent_edges:
        role = "mother" if people[parent].gender == "female" else "father"
        lines.append(f"{people[child].name}'s {role} is {people[parent].name}.")
    return lines


def _shuffle_choices(
    values: list[str], correct: str, seed: int, correct_position: int
) -> tuple[dict[str, str], str]:
    distractors = [value for value in values if value != correct]
    random.Random(seed).shuffle(distractors)
    shuffled = list(distractors)
    shuffled.insert(correct_position, correct)
    if len(shuffled) != len(LETTERS):
        raise ValueError("exactly four answer choices are required")
    choices = dict(zip(LETTERS, shuffled))
    correct_letter = next(letter for letter, value in choices.items() if value == correct)
    return choices, correct_letter


def _choice_text(choices: dict[str, str]) -> str:
    return ", ".join(f"{letter}. {value}" for letter, value in choices.items())


def render_concrete(
    problem: Problem, seed: int, correct_position: int, explicit: bool = False
) -> Trial:
    people = _person_map(problem)
    values = [problem.correct_relation, *RELATION_DISTRACTORS[problem.correct_relation]]
    choices, correct_letter = _shuffle_choices(
        values, problem.correct_relation, seed, correct_position
    )
    facts = " ".join(_fact_sentences(problem))
    definition = ""
    if explicit:
        term = problem.correct_relation
        definition = f" In this task, {_relation_definition(term)}."
    prompt = (
        f"{facts}{definition} What is {people[problem.target].name}'s relationship "
        f"to {people[problem.source].name}? {_choice_text(choices)}. "
        "Respond only with the answer letter."
    )
    condition = "explicit" if explicit else "concrete"
    return Trial(
        trial_id=f"{problem.problem_id}-{condition}", problem_id=problem.problem_id,
        condition=condition, prompt=prompt, choices=choices,
        correct_letter=correct_letter, correct_semantic_answer=problem.correct_relation,
        path_length=PATH_LENGTH[problem.correct_relation], distractor_count=len(problem.distractor_facts),
        prompt_chars=len(prompt),
    )


def render_arbitrary(problem: Problem, seed: int, correct_position: int) -> Trial:
    rng = random.Random(seed)
    labels = rng.sample(NOVEL, 4)
    mother, father, sibling, target_relation = labels
    people = _person_map(problem)
    entity = {person.id: f"e{i}" for i, person in enumerate(problem.people)}
    facts = []
    for child, parent in problem.parent_edges:
        rel = mother if people[parent].gender == "female" else father
        facts.append(f"{rel}({entity[child]},{entity[parent]})")
    gender = "female" if problem.correct_relation == "great-aunt" else "male"
    candidates = [entity[p.id] for p in problem.people if p.id not in {problem.source, problem.target}]
    values = [entity[problem.target], *candidates[:2], "None"]
    choices, correct_letter = _shuffle_choices(
        values, entity[problem.target], seed + 1, correct_position
    )
    prompt = (
        f"Facts: {', '.join(facts)}. Definitions: {mother}(x,y) means y is the female "
        f"parent of x; {father}(x,y) means y is the male parent of x. "
        f"{sibling}(x,y) means x and y share a parent and y is {gender}. "
        f"{_abstract_definition(problem.correct_relation, target_relation, sibling)}. "
        f"Which entity stands in relation {target_relation} to {entity[problem.source]}? "
        f"{_choice_text(choices)}. Respond only with the answer letter."
    )
    return Trial(
        trial_id=f"{problem.problem_id}-arbitrary", problem_id=problem.problem_id,
        condition="arbitrary", prompt=prompt, choices=choices,
        correct_letter=correct_letter, correct_semantic_answer=entity[problem.target],
        path_length=PATH_LENGTH[problem.correct_relation], distractor_count=len(problem.distractor_facts),
        prompt_chars=len(prompt),
    )


def render_formal(problem: Problem, seed: int, correct_position: int) -> Trial:
    people = _person_map(problem)
    entity = {person.id: f"e{i}" for i, person in enumerate(problem.people)}
    facts = []
    for child, parent in problem.parent_edges:
        rel = "M" if people[parent].gender == "female" else "F"
        facts.append(f"{rel}({entity[child]},{entity[parent]})")
    gender = "female" if problem.correct_relation == "great-aunt" else "male"
    candidates = [entity[p.id] for p in problem.people if p.id not in {problem.source, problem.target}]
    values = [entity[problem.target], *candidates[:2], "None"]
    choices, correct_letter = _shuffle_choices(
        values, entity[problem.target], seed + 2, correct_position
    )
    if problem.correct_relation in {"mother", "father"}:
        formula = "Define G(x,y) to hold when M(x,y) or F(x,y), with y having the required gender."
    elif problem.correct_relation in {"sister", "brother"}:
        formula = "Define G(x,y) to hold when S(x,y)."
    elif problem.correct_relation in {"grandmother", "grandfather"}:
        formula = "Define G(x,y) to hold when there exists u such that either M(x,u) or F(x,u), and either M(u,y) or F(u,y), with y having the required gender."
    elif problem.correct_relation in {"aunt", "uncle"}:
        formula = "Define G(x,y) to hold when there exists u such that either M(x,u) or F(x,u), and S(u,y)."
    else:
        formula = "Define G(x,y) to hold when there exist u and v such that either M(x,u) or F(x,u), either M(u,v) or F(u,v), and S(v,y)."
    prompt = (
        "Relations are directed from the first argument to the second. M(x,y) means y is "
        "x's female parent; F(x,y) means y is x's male parent. "
        f"Facts: {', '.join(facts)}. Define S(x,y) to mean x and y share a parent and y is "
        f"{gender}. {formula} "
        f"For which choice does G({entity[problem.source]},y) hold? {_choice_text(choices)}. "
        "Respond only with the answer letter."
    )
    return Trial(
        trial_id=f"{problem.problem_id}-formal", problem_id=problem.problem_id,
        condition="formal", prompt=prompt, choices=choices,
        correct_letter=correct_letter, correct_semantic_answer=entity[problem.target],
        path_length=PATH_LENGTH[problem.correct_relation], distractor_count=len(problem.distractor_facts),
        prompt_chars=len(prompt),
    )


def render_trials(problems: list[Problem], seed: int) -> list[Trial]:
    trials: list[Trial] = []
    for index, problem in enumerate(problems):
        base = seed + index * 17
        trials.extend([
            render_concrete(problem, base, index % 4),
            render_concrete(problem, base + 1, (index + 1) % 4, explicit=True),
            render_arbitrary(problem, base + 2, (index + 2) % 4),
            render_formal(problem, base + 3, (index + 3) % 4),
        ])
    random.Random(seed).shuffle(trials)
    return trials
