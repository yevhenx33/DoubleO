import os
import random
import json
import string

def generate_math_dataset(num_samples=10000):
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
        equation = f"{a} {op} {b} = {res}"
        data.append(equation)
    return data

def generate_code_dataset(num_samples=10000):
    data = []
    funcs = [
        ("def add(a, b):\n    return a + b", "add"),
        ("def sub(a, b):\n    return a - b", "sub"),
        ("def mul(a, b):\n    return a * b", "mul"),
        ("def greet(name):\n    return 'Hello ' + name", "greet"),
        ("def is_even(n):\n    return n % 2 == 0", "is_even"),
        ("def square(x):\n    return x * x", "square"),
        ("def double(n):\n    return n * 2", "double"),
        ("def get_first(lst):\n    return lst[0]", "get_first"),
    ]
    
    for _ in range(num_samples):
        func_str, name = random.choice(funcs)
        # Add random comments or variable renames to make it slightly diverse
        var_suffix = random.choice(string.ascii_lowercase)
        if random.random() < 0.5:
            # Replace 'a' with something else
            f = func_str.replace('(a, b)', f'(var_a_{var_suffix}, var_b_{var_suffix})')
            f = f.replace('a + b', f'var_a_{var_suffix} + var_b_{var_suffix}')
            f = f.replace('a - b', f'var_a_{var_suffix} - var_b_{var_suffix}')
            f = f.replace('a * b', f'var_a_{var_suffix} * var_b_{var_suffix}')
            data.append(f)
        else:
            data.append(func_str + f"\n    # Computed {name}")
    return data

def build_tokenizer(math_data, code_data):
    # Character-level tokenizer
    chars = set(''.join(math_data) + ''.join(code_data))
    # Add special tokens
    chars.add('<|endoftext|>')
    chars.add('\n')
    chars = sorted(list(chars))
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return stoi, itos, len(chars)

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(ids, itos):
    return ''.join([itos[i] for i in ids])

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating synthetic datasets...")
    math_data = generate_math_dataset(20000)
    code_data = generate_code_dataset(20000)
    
    print("Building tokenizer...")
    stoi, itos, vocab_size = build_tokenizer(math_data, code_data)
    
    vocab_meta = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "vocab_size": vocab_size
    }
    
    with open(os.path.join(script_dir, "vocab.json"), "w") as f:
        json.dump(vocab_meta, f)
        
    print(f"Vocab size: {vocab_size}")
    
    # Save raw datasets
    with open(os.path.join(script_dir, "math.jsonl"), "w") as f:
        for ex in math_data:
            f.write(json.dumps({"text": ex}) + "\n")
            
    with open(os.path.join(script_dir, "code.jsonl"), "w") as f:
        for ex in code_data:
            f.write(json.dumps({"text": ex}) + "\n")
            
    print("Saved datasets to data/")
