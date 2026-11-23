"""Prompt templates for zero-shot candidate verification with VideoRefer.

`[<region>]` is replaced by VideoRefer with the object-level mask tokens, the same
way as in the official VideoRefer-Bench inference code.
"""

YESNO_PROMPTS = {
    # p0: default, closest to the CRF-SAM3 Yes/No discriminator wording
    "p0": ('Is the object [<region>] in the video the target referred to by the expression: '
           '"{exp}"? Answer Yes or No.'),
    # p1: description-matching wording
    "p1": ('There is an object [<region>] in the video. Does the description "{exp}" '
           'correctly refer to this object? Please answer with Yes or No.'),
    # p2: reasoning-friendly wording (ReVOS reasoning expressions are often questions)
    "p2": ('Referring expression or question: "{exp}"\n'
           'Considering the whole video, is the object [<region>] the object that this '
           'expression refers to (or the answer to the question)? Answer Yes or No.'),
}

CHOICE_PROMPTS = {
    # c0: multiple-choice format, mirrors VideoRefer-Bench-Q
    # ("... Answer with the option's letter from the given choices directly.")
    "c0": ('The video contains {n} candidate objects: {options}. '
           'Which candidate is referred to by the expression "{exp}"? '
           "Answer with the option's letter from the given choices directly."),
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_yesno(prompt_id: str, expression: str) -> str:
    return YESNO_PROMPTS[prompt_id].format(exp=expression.strip())


def build_choice(prompt_id: str, expression: str, n: int) -> str:
    options = "; ".join(f"({LETTERS[i]}) [<region>]" for i in range(n))
    return CHOICE_PROMPTS[prompt_id].format(n=n, options=options, exp=expression.strip())
