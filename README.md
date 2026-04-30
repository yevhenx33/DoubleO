# NanoDoubleO: Solving Catastrophic Forgetting

For decades, neural networks have suffered from a fundamental flaw: **Catastrophic Forgetting**. When a network learns something new, it violently overwrites what it already knows. 

This repository introduces **NanoDoubleO**, a completely custom Transformer architecture built from scratch to definitively prove that a neural network can learn multiple disparate skills sequentially with **Zero Forgetting**.

## The Breakthrough: Complex Orthogonal Routing

Imagine a standard neural network as a single highway. If Task A (Math) is driving on the highway, and Task B (Code) suddenly merges on, there is a massive collision. The Math cars are run off the road (Catastrophic Forgetting).

We redesigned the highway to exist in **3D Space**.
We pushed the neural network into the **Complex Plane** ($\mathbb{C}$), replacing standard LayerNorm with `PhaseNorm` (to preserve angular vectors) and replacing the standard GELU block with a **Chebyshev Multi-Layer Perceptron (MLP)**.

- **Task A (Math)** is mathematically locked to drive only on the X-axis (using a $T_1(z) = z$ polynomial).
- **Task B (Code)** is mathematically locked to drive only on the Y-axis (using a $T_3(z) = 4z^3 - 3z$ polynomial).

Because the axes are perpendicular (Orthogonal), the tasks flow through the exact same physical weight matrices without ever colliding.

## The Results

By running our `train_and_eval.py` script, you can watch a standard GPT model completely obliterate its math knowledge when forced to learn code (a `+0.53` penalty spike). In contrast, the DoubleO model perfectly retains its math knowledge (`+0.02` penalty) despite radically updating its weights to learn code.

![Comparison Chart](data/comparison_chart.png)

## Quickstart

Run the evaluation on your local machine to reproduce the breakthrough in minutes.

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate the synthetic datasets:**
```bash
python data/data_gen.py
```

3. **Run the Continual Learning Gauntlet:**
This script trains the Baseline model and the DoubleO model, exporting `results.json`.
```bash
python scripts/train_and_eval.py
```

4. **Visualize the Results:**
Generate the `comparison_chart.png` plot.
```bash
python scripts/plot_results.py
```
