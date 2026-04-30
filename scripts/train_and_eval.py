import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_double_o import DoubleOGPT, DoubleOGPTConfig
from model_baseline import StandardGPT, StandardGPTConfig

class TextDataset(Dataset):
    def __init__(self, filepath, stoi, block_size):
        with open(filepath, 'r') as f:
            lines = [json.loads(line)['text'] for line in f]
        
        self.data = []
        for line in lines:
            tokens = [stoi.get(c, 0) for c in line]
            tokens.append(stoi['<|endoftext|>'])
            self.data.extend(tokens)
            
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

def evaluate(model, dataloader, task_idx, device, num_batches=20):
    model.eval()
    model.set_task(task_idx)
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches: break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def run_gauntlet(model_name, model, math_loader, code_loader, device):
    print(f"\n==========================================")
    print(f"RUNNING GAUNTLET FOR: {model_name}")
    print(f"==========================================\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    math_iter = iter(math_loader)
    code_iter = iter(code_loader)
    
    print("=== Phase 1: Pre-training Core Network (Mixture) ===")
    model.train()
    for step in range(250):
        task_idx = step % 2
        model.set_task(task_idx)
        try:
            x, y = next(math_iter) if task_idx == 0 else next(code_iter)
        except StopIteration:
            if task_idx == 0:
                math_iter = iter(math_loader)
                x, y = next(math_iter)
            else:
                code_iter = iter(code_loader)
                x, y = next(code_iter)
                
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"[{model_name}] Pre-train step {step}, loss: {loss.item():.4f}")
            
    math_pt = evaluate(model, math_loader, 0, device)
    code_pt = evaluate(model, code_loader, 1, device)
    
    print("\n=== Phase 2: Freezing Core Network ===")
    for name, param in model.named_parameters():
        if 'mlp' not in name:
            param.requires_grad = False
            
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    print("\n=== Phase 3: Train Task A (Math) ===")
    model.set_task(0)
    for step, (x, y) in enumerate(math_loader):
        if step >= 250: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    math_a = evaluate(model, math_loader, 0, device)
    code_a = evaluate(model, code_loader, 1, device)
    
    print("\n=== Phase 4: Train Task B (Code) ===")
    model.set_task(1)
    for step, (x, y) in enumerate(code_loader):
        if step >= 250: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    math_b = evaluate(model, math_loader, 0, device)
    code_b = evaluate(model, code_loader, 1, device)
    
    forgetting = math_b - math_a
    print(f"\nFinal Forgetting Penalty: {forgetting:.4f}")
    
    return {
        "math_loss": [math_pt, math_a, math_b],
        "code_loss": [code_pt, code_a, code_b],
        "forgetting": forgetting
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    with open(os.path.join(data_dir, "vocab.json"), 'r') as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    vocab_size = vocab['vocab_size']
    
    block_size = 32
    batch_size = 8
    
    math_dataset = TextDataset(os.path.join(data_dir, "math.jsonl"), stoi, block_size)
    code_dataset = TextDataset(os.path.join(data_dir, "code.jsonl"), stoi, block_size)
    math_loader = DataLoader(math_dataset, batch_size=batch_size, shuffle=True)
    code_loader = DataLoader(code_dataset, batch_size=batch_size, shuffle=True)
    
    # 1. Run StandardGPT Baseline
    std_config = StandardGPTConfig()
    std_config.vocab_size = vocab_size
    std_model = StandardGPT(std_config).to(device)
    res_std = run_gauntlet("StandardGPT", std_model, math_loader, code_loader, device)
    
    # 2. Run DoubleOGPT
    do_config = DoubleOGPTConfig()
    do_config.vocab_size = vocab_size
    do_model = DoubleOGPT(do_config).to(device)
    res_do = run_gauntlet("NanoDoubleO", do_model, math_loader, code_loader, device)
    
    results = {
        "StandardGPT": res_std,
        "NanoDoubleO": res_do
    }
    
    with open(os.path.join(data_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved complete evaluation results to data/results.json")

if __name__ == "__main__":
    main()
