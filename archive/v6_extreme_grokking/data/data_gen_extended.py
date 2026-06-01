"""
Extended data generator for the 4-task continual learning gauntlet.
Produces 50K samples per task across 4 structurally distinct domains.
"""
import os
import random
import json
import string

def generate_math_dataset(num_samples=50000):
    """Arithmetic expressions: '234 + 567 = 801'"""
    data = []
    for _ in range(num_samples):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            res = a + b
        elif op == '-':
            res = a - b
        else:
            res = a * b
        data.append(f"{a} {op} {b} = {res}")
    return data

def generate_code_dataset(num_samples=50000):
    """Python function snippets with variable renaming."""
    templates = [
        ("def add(a, b):\n    return a + b", "add"),
        ("def sub(a, b):\n    return a - b", "sub"),
        ("def mul(a, b):\n    return a * b", "mul"),
        ("def greet(name):\n    return 'Hello ' + name", "greet"),
        ("def is_even(n):\n    return n % 2 == 0", "is_even"),
        ("def square(x):\n    return x * x", "square"),
        ("def double(n):\n    return n * 2", "double"),
        ("def get_first(lst):\n    return lst[0]", "get_first"),
        ("def negate(x):\n    return -x", "negate"),
        ("def inc(x):\n    return x + 1", "inc"),
        ("def dec(x):\n    return x - 1", "dec"),
        ("def is_zero(x):\n    return x == 0", "is_zero"),
    ]
    data = []
    for _ in range(num_samples):
        func_str, name = random.choice(templates)
        vs = random.choice(string.ascii_lowercase)
        if random.random() < 0.5:
            f = func_str.replace('(a, b)', f'(v_{vs}_a, v_{vs}_b)')
            f = f.replace('a + b', f'v_{vs}_a + v_{vs}_b')
            f = f.replace('a - b', f'v_{vs}_a - v_{vs}_b')
            f = f.replace('a * b', f'v_{vs}_a * v_{vs}_b')
            data.append(f)
        else:
            data.append(func_str + f"\n    # {name}")
    return data

def generate_logic_dataset(num_samples=50000):
    """Boolean logic expressions: 'true and false = false'"""
    data = []
    ops = {
        'and': lambda a, b: a and b,
        'or': lambda a, b: a or b,
        'xor': lambda a, b: a ^ b,
    }
    for _ in range(num_samples):
        if random.random() < 0.25:
            # Unary NOT
            val = random.choice([True, False])
            result = not val
            expr = f"not {str(val).lower()} = {str(result).lower()}"
        else:
            a = random.choice([True, False])
            b = random.choice([True, False])
            op_name = random.choice(list(ops.keys()))
            result = ops[op_name](a, b)
            expr = f"{str(a).lower()} {op_name} {str(b).lower()} = {str(result).lower()}"
        data.append(expr)
    return data

def generate_spell_dataset(num_samples=50000):
    """Spelling patterns: 'c-a-t = cat', 'd-o-g = dog'"""
    words = [
        "cat", "dog", "sun", "run", "hat", "bat", "log", "fog",
        "pen", "hen", "net", "set", "cup", "pup", "bug", "rug",
        "top", "hop", "sit", "hit", "fan", "van", "jam", "ham",
        "pin", "tin", "dot", "hot", "mud", "bud", "nut", "cut",
        "red", "bed", "wet", "jet", "fox", "box", "mix", "six",
        "map", "tap", "dip", "tip", "pod", "rod", "gum", "sum",
        "wig", "pig", "dim", "rim", "cob", "job", "fun", "gun",
    ]
    data = []
    for _ in range(num_samples):
        w = random.choice(words)
        spelled = "-".join(list(w))
        data.append(f"{spelled} = {w}")
    return data

def build_tokenizer(*datasets):
    chars = set()
    for ds in datasets:
        chars.update(set(''.join(ds)))
    chars.add('<|endoftext|>')
    chars.add('\n')
    chars = sorted(list(chars))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)

if __name__ == "__main__":
    random.seed(42)  # Fixed seed for reproducibility
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extended")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating extended datasets (50K samples × 4 tasks)...")
    math_data = generate_math_dataset(50000)
    code_data = generate_code_dataset(50000)
    logic_data = generate_logic_dataset(50000)
    spell_data = generate_spell_dataset(50000)

    print("Building tokenizer...")
    stoi, itos, vocab_size = build_tokenizer(math_data, code_data, logic_data, spell_data)
    print(f"Vocab size: {vocab_size}")

    vocab_meta = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "vocab_size": vocab_size
    }
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab_meta, f)

    task_names = ["math", "code", "logic", "spell"]
    task_data = [math_data, code_data, logic_data, spell_data]
    for name, ds in zip(task_names, task_data):
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for ex in ds:
                f.write(json.dumps({"text": ex}) + "\n")
        print(f"  {name}.jsonl: {len(ds)} samples")

    print(f"All datasets saved to {out_dir}/")
