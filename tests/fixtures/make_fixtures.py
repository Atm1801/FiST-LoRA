"""Regenerate the tiny synthetic datasets used by the smoke tests (deterministic)."""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
WORDS = ("the cat dog sat ran on a mat park big small red blue quickly slowly happy sad "
         "tree house car road river sun moon star bird fish").split()


def sentence(rng, n=8):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, n)))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    rng = random.Random(0)
    for split, n in (("train", 64), ("validation", 32)):
        rows = []
        for _ in range(n):
            s1 = sentence(rng)
            same = rng.random() < 0.5
            rows.append({"sentence1": s1, "sentence2": s1 if same else sentence(rng), "label": int(not same)})
        write_jsonl(HERE / "glue_pair" / f"{split}.jsonl", rows)
        write_jsonl(HERE / "glue_regression" / f"{split}.jsonl",
                    [{"sentence1": r["sentence1"], "sentence2": r["sentence2"], "label": round(rng.uniform(0, 5), 2)} for r in rows])
    cs = []
    for _ in range(48):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        cs.append({"instruction": f"Is {a} larger than {b}? Answer format: true/false",
                   "input": "", "output": f"the correct answer is {str(a > b).lower()}", "answer": str(a > b).lower()})
    with open(HERE / "commonsense_mini.json", "w") as f:
        json.dump(cs, f)
    math_train, gsm, math_test = [], [], []
    for _ in range(48):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        math_train.append({"query": f"What is {a} plus {b}?", "response": f"{a} + {b} = {a + b}. The answer is: {a + b}"})
    for _ in range(6):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        gsm.append({"question": f"What is {a} plus {b}?", "answer": f"{a} + {b} = {a + b}\n#### {a + b}"})
        math_test.append({"problem": f"Compute ${a}+{b}$.", "solution": f"We get $\\boxed{{{a + b}}}$."})
    write_jsonl(HERE / "metamath_mini.jsonl", math_train)
    write_jsonl(HERE / "gsm8k_mini.jsonl", gsm)
    write_jsonl(HERE / "math_mini.jsonl", math_test)


if __name__ == "__main__":
    main()
