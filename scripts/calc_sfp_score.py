import json

# Define the metric calculation
def calc_retention(L_A_init, L_A_mid, L_A_final):
    # R = (L_A_init - L_A_final) / (L_A_init - L_A_mid)
    # Bounded between 0 and 1
    r = (L_A_init - L_A_final) / (L_A_init - L_A_mid + 1e-9)
    return max(0.0, min(1.0, r))

def calc_plasticity(L_B_init, L_B_final, L_B_optimal):
    # P = (L_B_init - L_B_final) / (L_B_init - L_B_optimal)
    # Bounded between 0 and 1
    p = (L_B_init - L_B_final) / (L_B_init - L_B_optimal + 1e-9)
    return max(0.0, min(1.0, p))

def load_metrics(filepath, model_key):
    with open(filepath, 'r') as f:
        data = json.load(f)
    metrics = data[model_key]["metrics"]
    
    # Task A
    val_A = metrics["val_A"]
    L_A_init = val_A[0][1] # Step 0
    L_A_mid = val_A[40][1] # Step 2000 (40 * 50)
    L_A_final = val_A[-1][1] # Step 4000
    
    # Task B
    val_B = metrics["val_B"]
    L_B_init = val_B[0][1] # Step 2000
    L_B_final = val_B[-1][1] # Step 4000
    
    return {
        "L_A_init": L_A_init,
        "L_A_mid": L_A_mid,
        "L_A_final": L_A_final,
        "L_B_init": L_B_init,
        "L_B_final": L_B_final
    }

# Load all models
nano = load_metrics('data/continual_learning_results.json', 'nanogpt')
massive = load_metrics('data/continual_learning_results.json', 'massive_pure')
poly = load_metrics('data/poly_moe_results.json', 'poly_moe')
ewc = load_metrics('data/ewc_continual_results.json', 'ewc_pure')
cheby = load_metrics('data/cheby_moe_results.json', 'cheby_moe')

models = {
    "NanoGPT": nano,
    "Massive Pure": massive,
    "Poly-MoE": poly,
    "EWC + Prototype": ewc,
    "Cheby-MoE": cheby
}

# Define L_B_optimal as NanoGPT's final loss on Task B
L_B_optimal = nano["L_B_final"]

print(f"{'Model':<20} | {'Retention (R)':<15} | {'Plasticity (P)':<15} | {'SFP Score':<10}")
print("-" * 65)

for name, m in models.items():
    R = calc_retention(m["L_A_init"], m["L_A_mid"], m["L_A_final"])
    P = calc_plasticity(m["L_B_init"], m["L_B_final"], L_B_optimal)
    
    if R + P > 0:
        SFP = 2 * (R * P) / (R + P)
    else:
        SFP = 0.0
        
    print(f"{name:<20} | {R:15.4f} | {P:15.4f} | {SFP:10.4f}")
