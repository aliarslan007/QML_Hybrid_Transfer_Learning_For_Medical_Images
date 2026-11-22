"""Classification heads. Each maps d input angles to one logit (pneumonia score).

Parameter counts (d = 4 / d = 8):
  LinearHead            d + 1                  ->  5 /  9
  MLPHead (h = 4 / 5)   (d + 2) h + 1          -> 25 / 51
  VQCHead (L = 2)       3 d L + 2              -> 26 / 50
"""
import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VQCHead(nn.Module):
    """Variational quantum circuit head (PennyLane, simulated).

    Circuit on d qubits, all starting in |0>:
      1. Encoding:  RY(x_i) on qubit i                       (angle encoding)
      2. L layers:  RY(a) RZ(b) RY(c) on every qubit          (3 d L trainable angles)
                    then CNOT ring 0->1->...->d-1->0          (entanglement; optional)
      3. Readout:   z = mean_i <Z_i>,  logit = scale * z + bias  (2 trainable numbers)

    Why the mean of all <Z_i> and not only <Z_0>: without entanglement,
    <Z_0> would depend on x_0 alone, making the no-entanglement ablation
    degenerate. Averaging over qubits keeps every feature in play for both
    variants, at no extra parameters.
    """

    def __init__(self, d: int, layers: int = 2, entangle: bool = True,
                 diff_method: str = "backprop", init_std: float = 0.1,
                 device_name: str = "default.qubit"):
        super().__init__()
        import pennylane as qml

        self.d, self.layers, self.entangle = d, layers, entangle
        self.weights = nn.Parameter(init_std * torch.randn(layers, d, 3))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

        dev = qml.device(device_name, wires=d)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, w):
            # x has shape (batch, d): PennyLane broadcasts over the batch
            for i in range(d):
                qml.RY(x[..., i], wires=i)
            for l in range(layers):
                for i in range(d):
                    qml.RY(w[l, i, 0], wires=i)
                    qml.RZ(w[l, i, 1], wires=i)
                    qml.RY(w[l, i, 2], wires=i)
                if entangle and d > 1:
                    for i in range(d):
                        qml.CNOT(wires=[i, (i + 1) % d])
            return [qml.expval(qml.PauliZ(i)) for i in range(d)]

        self.circuit = circuit
        self.qml = qml

    def expvals(self, x):
        """<Z_i> for every qubit, shape (batch, d)."""
        out = self.circuit(x, self.weights)
        return torch.stack([o.reshape(-1) for o in out], dim=-1).to(x.dtype)

    def forward(self, x):
        z = self.expvals(x).mean(dim=-1)
        return self.scale * z + self.bias

    def draw(self) -> str:
        x = torch.zeros(self.d)
        return self.qml.draw(self.circuit)(x, self.weights.detach())


def build_head(name: str, d: int, cfg: dict) -> nn.Module:
    if name == "linear":
        return LinearHead(d)
    if name == "mlp":
        return MLPHead(d, cfg["mlp_hidden"][d])
    if name.startswith("vqc"):
        v = cfg["vqc"]
        layers = int(name.split("_L")[1]) if "_L" in name else v["layers"]
        return VQCHead(d, layers=layers,
                    entangle=(name != "vqc_noent") and v["entangle"],
                    diff_method=v["diff_method"], init_std=v["init_std"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
