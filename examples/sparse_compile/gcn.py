import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        # Two-layer GCN.
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, out_size)

    def forward(self, A_norm: dglsp.SparseMatrix, X: torch.Tensor):
        X = dglsp.spmm(A_norm, self.W1(X))
        X = F.relu(X)
        X = dglsp.spmm(A_norm, self.W2(X))
        return X


model = GCN(10, 20)
try:
    scripted_model = torch.jit.script(model)
    print(f"GCN is jittable. The code is:\n{scripted_model.code}")
except Exception as e:
    print(f"GCN is not jittable.")


try:
    traced_model: torch.fx.GraphModule = symbolic_trace(model)
    print(f"GCN is traceable. The code is:\n{traced_model.code}")
except Exception as e:
    print(f"GCN is not traceable")
