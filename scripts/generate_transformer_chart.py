import json
import matplotlib.pyplot as plt
import seaborn as sns

def generate_chart():
    with open('data/transformer_moe_results.json', 'r') as f:
        data = json.load(f)
        
    metrics = data['transformer_moe']['metrics']
    steps_a = [m[0] for m in metrics['val_A']]
    vals_a = [m[1] for m in metrics['val_A']]
    
    steps_b = [m[0] for m in metrics['val_B']]
    vals_b = [m[1] for m in metrics['val_B']]
    
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps_a, vals_a, color="#3498db", label="Task A (Shakespeare)", linewidth=2.5)
    ax.plot(steps_b, vals_b, color="#e74c3c", label="Task B (Math)", linewidth=2.5)
    
    ax.axvline(x=2000, color="white", linestyle="--", alpha=0.5)
    ax.text(2050, 6.0, "Phase 2 Begins\n(Expert 1 Frozen)", color="white", alpha=0.8)
    
    ax.set_title("Transformer Hub-and-Spoke (Shared Embeddings Failure)", fontsize=16, color="white", pad=20)
    ax.set_xlabel("Training Steps", fontsize=12, color="white")
    ax.set_ylabel("Validation Loss", fontsize=12, color="white")
    
    ax.set_ylim(1.0, 10.0)
    
    ax.legend(facecolor='#2C3E50', edgecolor='none', labelcolor='white')
    
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#2C3E50')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('transformer_moe_chart.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print("Saved transformer_moe_chart.png")

if __name__ == "__main__":
    generate_chart()
