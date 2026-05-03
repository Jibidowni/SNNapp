from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate, utils
from tqdm import tqdm


# Input per timestep for N-MNIST frames is 2 * 34 * 34 = 2312
INPUT_FEATURES = 2312


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NMNISTMLPNet(nn.Module):
    """Simple feed-forward spiking MLP: 2312 -> 64 -> 16 -> 10."""

    def __init__(self, hidden1=64, hidden2=16, beta=0.5, spike_grad=None):
        super().__init__()
        spike_grad = spike_grad or surrogate.atan()

        self.fc1 = nn.Linear(INPUT_FEATURES, hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, reset_mechanism="zero")

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, reset_mechanism="zero")

        self.fc3 = nn.Linear(hidden2, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, output=True, reset_mechanism="zero")

        self._reset_layers = nn.ModuleList([self.lif1, self.lif2, self.lif3])

    def forward(self, data):
        """Forward through time.

        data shape: [time, batch, channels, h, w]
        returns: [time, batch, classes]
        """
        utils.reset(self._reset_layers)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        for step in range(data.size(0)):
            x = data[step].flatten(1)  # [batch, 2312]

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_rec.append(spk3)

        return torch.stack(spk_rec)

    def forward_record(self, data):
        """Forward pass with spike/membrane recordings for interpretability.

        returns: (spk_out, recordings)
        spk_out: [time, batch, classes]
        recordings:
            - hidden1, hidden2, output
            - each entry has tensors [time, batch, neurons]
        """
        utils.reset(self._reset_layers)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []

        for step in range(data.size(0)):
            x = data[step].flatten(1)

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1.detach().cpu())
            mem1_rec.append(mem1.detach().cpu())
            spk2_rec.append(spk2.detach().cpu())
            mem2_rec.append(mem2.detach().cpu())
            spk3_rec.append(spk3.detach().cpu())
            mem3_rec.append(mem3.detach().cpu())

        recordings = {
            "hidden1": {"spk": torch.stack(spk1_rec), "mem": torch.stack(mem1_rec)},
            "hidden2": {"spk": torch.stack(spk2_rec), "mem": torch.stack(mem2_rec)},
            "output": {"spk": torch.stack(spk3_rec), "mem": torch.stack(mem3_rec)},
        }

        return torch.stack(spk3_rec), recordings


def train_epoch(model, trainloader, optimizer, loss_fn, device, max_iters=None):
    loss_hist = []
    acc_hist = []
    total_iters = len(trainloader) if max_iters is None else min(max_iters, len(trainloader))

    model.train()
    for iteration, (data, targets) in enumerate(
        tqdm(trainloader, total=total_iters, desc="Train", leave=False)
    ):
        data = data.to(device)
        targets = targets.to(device)

        spk_rec = model(data)
        loss_val = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        acc = SF.accuracy_rate(spk_rec, targets)
        loss_hist.append(loss_val.item())
        acc_hist.append(acc)

        if max_iters is not None and iteration + 1 >= max_iters:
            break

    return loss_hist, acc_hist


def test_epoch(model, testloader, loss_fn, device, max_iters=None):
    loss_hist = []
    acc_hist = []
    total_iters = len(testloader) if max_iters is None else min(max_iters, len(testloader))

    model.eval()
    with torch.no_grad():
        for iteration, (data, targets) in enumerate(
            tqdm(testloader, total=total_iters, desc="Test", leave=False)
        ):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = model(data)
            loss_val = loss_fn(spk_rec, targets)

            acc = SF.accuracy_rate(spk_rec, targets)
            loss_hist.append(loss_val.item())
            acc_hist.append(acc)

            if max_iters is not None and iteration + 1 >= max_iters:
                break

    return loss_hist, acc_hist


def save_weights(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def load_model(weights_path, device=None, beta=0.5):
    device = device or get_device()
    model = NMNISTMLPNet(beta=beta).to(device)
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_sample(model, sample, device=None, record=True):
    """Run inference on a single sample.

    sample shape: [time, batch=1, channels, h, w]
    """
    device = device or get_device()
    model.to(device)

    with torch.no_grad():
        sample = sample.to(device)
        if record and hasattr(model, "forward_record"):
            spk_rec, recordings = model.forward_record(sample)
        else:
            spk_rec = model(sample)
            recordings = None

        summed = spk_rec.sum(dim=0).squeeze(0)  # [classes]
        pred = int(torch.argmax(summed).item())

    return pred, recordings
