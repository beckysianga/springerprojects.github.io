
# ============================================================
# Google Colab-ready implementation
# Personalised Federated Learning with PR-AS-HDP / RDP Accounting
# Dataset: Diabetes Health Indicators
# ============================================================

import copy
import json
import math
import os
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ============================================================
# 1. Utilities
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def safe_tensor(x: torch.Tensor, clip_val: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clip_val, neginf=-clip_val)


# ============================================================
# 2. Configuration
# ============================================================

@dataclass
class Config:
    csv_path: str = "/content/diabetes.csv"
    target_col: str = "Diabetes_binary"
    positive_label: int = 1

    test_size: float = 0.10
    val_size: float = 0.10

    n_clients: int = 8
    rounds: int = 30
    client_frac: float = 0.75
    local_epochs: int = 2
    batch_size: int = 128
    lr_encoder: float = 4e-4
    lr_head: float = 7e-4
    weight_decay: float = 1e-5
    dirichlet_alpha: float = 0.5

    hidden_dim1: int = 128
    hidden_dim2: int = 64
    rep_dim: int = 32
    head_hidden: int = 32
    dropout: float = 0.10
    use_batchnorm: bool = True

    personal_encoder_blend: float = 0.20

    lambda1: float = 0.20
    lambda2: float = 0.20
    lambda3: float = 0.05
    sensitivity_cap: float = 250.0
    sensitivity_floor: float = 1e-6

    alpha_ema: float = 0.90
    quantile_q: float = 0.80
    cmin: float = 0.50
    cmax: float = 6.00
    initial_r: float = 1.60

    eps_base_local: float = 0.80
    eps_base_central: float = 0.60
    eps_min: float = 0.15
    eps_max: float = 2.50
    beta_sched: float = 0.04
    gamma_imbalance: float = 0.25
    warmup_rounds: int = 8
    fixed_warmup_eps: float = 0.90

    delta: float = 1e-5
    rdp_orders: Tuple[float, ...] = (1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128)
    max_noise_scale: float = 10.0

    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    patience: int = 8
    min_delta: float = 1e-4

    seeds: Tuple[int, ...] = (42, 52, 62)
    output_dir: str = "/content/pr_ashdp_rdp_outputs"

CFG = Config()


# ============================================================
# 3. Data loading and partitioning
# ============================================================

def load_diabetes_csv(csv_path: str, target_col: str, positive_label: int = 1):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Upload diabetes.csv to /content or change cfg.csv_path.")
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = y.fillna(0)
    X = X.to_numpy(dtype=np.float32)
    y = (y.to_numpy() == positive_label).astype(np.float32)
    return safe_np(X).astype(np.float32), safe_np(y).astype(np.float32)



def split_data(X: np.ndarray, y: np.ndarray, cfg: Config, seed: int):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=seed,
        stratify=y,
    )
    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_trainval,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return safe_np(X_train), safe_np(y_train), safe_np(X_val), safe_np(y_val), safe_np(X_test), safe_np(y_test), scaler


def dirichlet_partition(X: np.ndarray, y: np.ndarray, n_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_indices = [np.where(y == c)[0] for c in classes]
    client_indices = [[] for _ in range(n_clients)]

    for idx in class_indices:
        idx = idx.copy()
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for cid, sp in enumerate(splits):
            client_indices[cid].extend(sp.tolist())

    clients = []
    all_idx = np.arange(len(y))
    for idxs in client_indices:
        idxs = np.array(idxs, dtype=int)
        if len(idxs) == 0:
            idxs = rng.choice(all_idx, size=1, replace=False)
        rng.shuffle(idxs)
        clients.append((safe_np(X[idxs]).astype(np.float32), safe_np(y[idxs]).astype(np.float32)))
    return clients


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    effective_bs = max(1, min(batch_size, len(ds)))
    drop_last = bool(shuffle and len(ds) > 1 and (len(ds) % effective_bs == 1))
    return DataLoader(ds, batch_size=effective_bs, shuffle=shuffle, drop_last=drop_last)


# ============================================================
# 4. Model
# ============================================================

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()

        def maybe_norm(dim: int):
            return nn.BatchNorm1d(dim) if cfg.use_batchnorm else nn.LayerNorm(dim)

        layers = [nn.Linear(input_dim, cfg.hidden_dim1), maybe_norm(cfg.hidden_dim1)]
        layers += [nn.ReLU(), nn.Dropout(cfg.dropout)]
        layers += [nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2), maybe_norm(cfg.hidden_dim2)]
        layers += [nn.ReLU(), nn.Dropout(cfg.dropout)]
        layers += [nn.Linear(cfg.hidden_dim2, cfg.rep_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PersonalHead(nn.Module):
    def __init__(self, rep_dim: int, head_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, z):
        return self.net(z)


class PersonalisedClientModel(nn.Module):
    def __init__(self, encoder: SharedEncoder, head: PersonalHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z


class InferenceModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z


def make_inference_model(encoder: nn.Module, head: nn.Module):
    return InferenceModel(copy.deepcopy(encoder), copy.deepcopy(head)).to(DEVICE)


# ============================================================
# 5. Loss and metrics
# ============================================================

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = safe_tensor(logits)
        targets = targets.float().view(-1, 1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.clamp(torch.sigmoid(logits), min=1e-6, max=1 - 1e-6)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


@torch.no_grad()
def predict_probs_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.array([])
    xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    logits, _ = model(xt)
    probs = torch.sigmoid(safe_tensor(logits)).detach().cpu().numpy().reshape(-1)
    return np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    probs = np.asarray(probs).reshape(-1)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.10, 0.90, 81):
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def evaluate_probs(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    y = np.asarray(safe_np(y)).reshape(-1).astype(int)
    probs = np.asarray(safe_np(probs)).reshape(-1)
    preds = (probs >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auprc": float(average_precision_score(y, probs)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, probs))
    except Exception:
        out["auroc"] = float("nan")
    return out


# ============================================================
# 6. Parameter helpers and adaptive mechanisms
# ============================================================

def flatten_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def set_params_from_flat(module: nn.Module, flat: torch.Tensor):
    ptr = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[ptr:ptr + n].view_as(p))
        ptr += n


def get_update(local_module: nn.Module, global_module: nn.Module) -> torch.Tensor:
    return flatten_params(local_module) - flatten_params(global_module)


def l2norm(t: torch.Tensor) -> float:
    return float(torch.norm(t, p=2).item())


def blend_encoder(local_encoder: nn.Module, global_encoder: nn.Module, strength: float):
    with torch.no_grad():
        for p_l, p_g in zip(local_encoder.parameters(), global_encoder.parameters()):
            delta = safe_tensor(p_l.data - p_g.data, clip_val=50.0)
            p_l.data = safe_tensor(p_g.data + strength * delta, clip_val=50.0)


def class_proportions(y: np.ndarray) -> Dict[int, float]:
    total = max(len(y), 1)
    return {int(c): float((y == c).sum()) / total for c in np.unique(y)}


def imbalance_factor(y: np.ndarray, eps: float = 1e-8) -> float:
    props = class_proportions(y)
    if not props:
        return 1.0
    k = len(props)
    return float(sum(1.0 / (props[c] + eps) for c in props) / k)


@torch.no_grad()
def representation_dispersion(encoder: nn.Module, X: np.ndarray) -> float:
    if len(X) == 0:
        return 0.0
    encoder.eval()
    xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    z = safe_tensor(encoder(xt), clip_val=50.0)
    std_vec = safe_tensor(torch.std(z, dim=0), clip_val=50.0)
    return float(torch.norm(std_vec, p=2).item())


def adaptive_clipping_threshold(prev_r: float, norms: List[float], cfg: Config):
    if len(norms) == 0:
        c_t = max(cfg.cmin, prev_r)
        return prev_r, c_t
    norms = np.nan_to_num(np.array(norms, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    qv = float(np.quantile(norms, cfg.quantile_q))
    r_t = cfg.alpha_ema * prev_r + (1.0 - cfg.alpha_ema) * qv
    c_t = float(np.clip(r_t, cfg.cmin, cfg.cmax))
    return float(r_t), c_t


def clip_update(update: torch.Tensor, clip_norm: float):
    norm = torch.norm(update, p=2)
    scale = min(1.0, clip_norm / (float(norm.item()) + 1e-12))
    return safe_tensor(update * scale, clip_val=50.0), float(norm.item())


def adaptive_sensitivity(dispersion: float, clipped_norm: float, imb: float, cfg: Config):
    s = 1.0 + cfg.lambda1 * dispersion + cfg.lambda2 * clipped_norm + cfg.lambda3 * imb
    s = float(np.clip(s, cfg.sensitivity_floor, cfg.sensitivity_cap))
    return s


def schedule_epsilon(round_idx: int, loss_proxy: float, imb: float, cfg: Config, base_eps: float):
    if round_idx <= cfg.warmup_rounds:
        return float(cfg.fixed_warmup_eps)
    factor = 1.0 + cfg.beta_sched * max(loss_proxy, 0.0) + cfg.gamma_imbalance * math.log1p(max(imb, 0.0))
    eps = base_eps * factor
    return float(np.clip(eps, cfg.eps_min, cfg.eps_max))


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float):
    epsilon = max(float(epsilon), 1e-12)
    delta = max(float(delta), 1e-12)
    return float(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon)


def laplace_noise_like(t: torch.Tensor, scale: float):
    if scale <= 0:
        return torch.zeros_like(t)
    dist = torch.distributions.Laplace(torch.tensor(0.0, device=t.device), torch.tensor(float(scale), device=t.device))
    return dist.sample(t.shape)


def add_laplace_noise(update: torch.Tensor, scale: float, cfg: Config):
    scale = float(np.clip(scale, 0.0, cfg.max_noise_scale))
    return safe_tensor(update + laplace_noise_like(update, scale), clip_val=50.0)


def add_gaussian_noise(update: torch.Tensor, sigma: float, cfg: Config):
    sigma = float(np.clip(sigma, 0.0, cfg.max_noise_scale))
    return safe_tensor(update + torch.randn_like(update) * sigma, clip_val=50.0)


# ============================================================
# 7. RDP accountant
# ============================================================

class RDPAccountant:
    def __init__(self, orders: Tuple[float, ...], delta: float):
        self.orders = np.array(orders, dtype=np.float64)
        self.delta = float(delta)
        self.rdp = np.zeros_like(self.orders, dtype=np.float64)
        self.events = []

    def add_gaussian(self, sensitivity: float, sigma: float, name: str):
        if sigma <= 0:
            return
        inc = self.orders * (float(sensitivity) ** 2) / (2.0 * (float(sigma) ** 2))
        self.rdp += inc
        self.events.append({"mechanism": name, "sensitivity": float(sensitivity), "sigma": float(sigma)})

    def epsilon(self):
        eps = self.rdp + np.log(1.0 / self.delta) / (self.orders - 1.0)
        idx = int(np.argmin(eps))
        return float(eps[idx]), float(self.orders[idx])


# ============================================================
# 8. Federated training
# ============================================================

def train_local_client(global_encoder, client_head, Xc, yc, cfg: Config):
    local_encoder = copy.deepcopy(global_encoder).to(DEVICE)
    local_head = copy.deepcopy(client_head).to(DEVICE)
    model = PersonalisedClientModel(local_encoder, local_head).to(DEVICE)
    model.train()

    criterion = BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": model.head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    loader = make_loader(Xc, yc, cfg.batch_size, shuffle=True)
    losses = []
    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

    blend_encoder(local_encoder, global_encoder, cfg.personal_encoder_blend)
    return local_encoder, local_head, float(np.mean(losses)) if losses else 0.0


def aggregate_updates(global_encoder, client_updates: List[torch.Tensor], weights: List[float]):
    base = flatten_params(global_encoder).detach().clone()
    agg_update = torch.zeros_like(base)
    for u, w in zip(client_updates, weights):
        agg_update += float(w) * u
    new_flat = safe_tensor(base + agg_update, clip_val=50.0)
    set_params_from_flat(global_encoder, new_flat)


def evaluate_personalised(global_encoder, client_heads, X, y, cfg: Config):
    probs_list = []
    for head in client_heads:
        model = make_inference_model(global_encoder, head)
        probs_list.append(predict_probs_model(model, X))
    probs = np.mean(np.vstack(probs_list), axis=0)
    return probs


def run_method(method_name: str, X_train, y_train, X_val, y_val, X_test, y_test, cfg: Config, seed: int):
    seed_everything(seed)
    input_dim = X_train.shape[1]
    global_encoder = SharedEncoder(input_dim, cfg).to(DEVICE)
    client_heads = [PersonalHead(cfg.rep_dim, cfg.head_hidden, cfg.dropout).to(DEVICE) for _ in range(cfg.n_clients)]
    clients = dirichlet_partition(X_train, y_train, cfg.n_clients, cfg.dirichlet_alpha, seed)

    accountant = RDPAccountant(cfg.rdp_orders, cfg.delta)
    r_state = cfg.initial_r
    best_val = -1.0
    best_encoder = copy.deepcopy(global_encoder.state_dict())
    best_heads = [copy.deepcopy(h.state_dict()) for h in client_heads]
    no_improve = 0
    history = []

    for rnd in range(1, cfg.rounds + 1):
        rng = np.random.default_rng(seed + rnd)
        m = max(1, int(math.ceil(cfg.client_frac * cfg.n_clients)))
        selected = sorted(rng.choice(np.arange(cfg.n_clients), size=m, replace=False).tolist())

        local_encoders = []
        local_heads = []
        raw_updates = []
        raw_norms = []
        local_losses = []
        client_sizes = []
        client_stats = []

        for cid in selected:
            Xc, yc = clients[cid]
            local_encoder, local_head, loss_val = train_local_client(global_encoder, client_heads[cid], Xc, yc, cfg)
            update = get_update(local_encoder, global_encoder).detach()
            raw_norm = l2norm(update)
            raw_updates.append(update)
            raw_norms.append(raw_norm)
            local_encoders.append(local_encoder)
            local_heads.append(local_head)
            local_losses.append(loss_val)
            client_sizes.append(len(yc))
            client_stats.append((representation_dispersion(local_encoder, Xc), imbalance_factor(yc)))

        if method_name == "NO_DP":
            clip_norm = max(raw_norms) if raw_norms else cfg.cmax
        elif method_name in ["FIXED_HDP"]:
            clip_norm = cfg.initial_r
        else:
            r_state, clip_norm = adaptive_clipping_threshold(r_state, raw_norms, cfg)

        noisy_updates = []
        clipped_norms = []
        local_eps_values = []
        central_eps_values = []
        sensitivities = []

        for update, (disp, imb), loss_val in zip(raw_updates, client_stats, local_losses):
            clipped_update, clipped_norm = clip_update(update, clip_norm)
            clipped_norms.append(clipped_norm)

            if method_name in ["A3_Clip+Budget+ClientSens", "A4_Full_PR_AS_HDP"]:
                sens = adaptive_sensitivity(disp, clipped_norm, imb, cfg)
            else:
                sens = 1.0
            sensitivities.append(sens)

            if method_name == "NO_DP":
                eps_l = 0.0
                noisy_update = clipped_update
            elif method_name in ["A2_Clip+Budget", "A3_Clip+Budget+ClientSens", "A4_Full_PR_AS_HDP"]:
                eps_l = schedule_epsilon(rnd, loss_val, imb, cfg, cfg.eps_base_local)
                scale = (sens * clip_norm) / max(eps_l, 1e-12)
                noisy_update = add_laplace_noise(clipped_update, scale, cfg)
            else:
                eps_l = cfg.fixed_warmup_eps
                scale = (sens * clip_norm) / max(eps_l, 1e-12)
                noisy_update = add_laplace_noise(clipped_update, scale, cfg)

            noisy_updates.append(noisy_update)
            local_eps_values.append(eps_l)

        sizes = np.array(client_sizes, dtype=np.float64)
        weights = sizes / max(sizes.sum(), 1.0)
        agg_before_server = torch.zeros_like(noisy_updates[0])
        for u, w in zip(noisy_updates, weights):
            agg_before_server += float(w) * u

        if method_name == "NO_DP":
            final_update = agg_before_server
            eps_c = 0.0
            sigma = 0.0
            server_sens = 0.0
        else:
            mean_loss = float(np.mean(local_losses)) if local_losses else 0.0
            mean_imb = float(np.mean([x[1] for x in client_stats])) if client_stats else 1.0
            if method_name in ["A2_Clip+Budget", "A3_Clip+Budget+ClientSens", "A4_Full_PR_AS_HDP"]:
                eps_c = schedule_epsilon(rnd, mean_loss, mean_imb, cfg, cfg.eps_base_central)
            else:
                eps_c = cfg.fixed_warmup_eps

            server_sens = float(np.mean(sensitivities)) * clip_norm
            if method_name == "A4_Full_PR_AS_HDP" or method_name == "FIXED_HDP":
                sigma = gaussian_sigma(server_sens, eps_c, cfg.delta)
                final_update = add_gaussian_noise(agg_before_server, sigma, cfg)
                accountant.add_gaussian(server_sens, max(sigma, 1e-12), f"round_{rnd}_server_gaussian")
            else:
                sigma = 0.0
                final_update = agg_before_server

        aggregate_updates(global_encoder, [final_update], [1.0])
        for cid, local_head in zip(selected, local_heads):
            client_heads[cid] = copy.deepcopy(local_head).to(DEVICE)

        val_probs = evaluate_personalised(global_encoder, client_heads, X_val, y_val, cfg)
        thr, _ = find_best_threshold(y_val, val_probs)
        val_metrics = evaluate_probs(y_val, val_probs, thr)
        eps_rdp, best_order = accountant.epsilon()

        history.append({
            "round": rnd,
            "method": method_name,
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "threshold": thr,
            "clip_norm": clip_norm,
            "mean_raw_norm": float(np.mean(raw_norms)) if raw_norms else 0.0,
            "mean_clipped_norm": float(np.mean(clipped_norms)) if clipped_norms else 0.0,
            "mean_sensitivity": float(np.mean(sensitivities)) if sensitivities else 0.0,
            "mean_eps_local": float(np.mean(local_eps_values)) if local_eps_values else 0.0,
            "eps_central": float(eps_c) if method_name != "NO_DP" else 0.0,
            "sigma": float(sigma),
            "rdp_epsilon": float(eps_rdp),
            "rdp_order": float(best_order),
        })

        if val_metrics["auprc"] > best_val + cfg.min_delta:
            best_val = val_metrics["auprc"]
            best_encoder = copy.deepcopy(global_encoder.state_dict())
            best_heads = [copy.deepcopy(h.state_dict()) for h in client_heads]
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Seed {seed} | {method_name} | Round {rnd:02d} | "
            f"Val AUPRC={val_metrics['auprc']:.4f} | F1={val_metrics['f1']:.4f} | "
            f"C={clip_norm:.3f} | RDP eps={eps_rdp:.3f}"
        )

        if no_improve >= cfg.patience:
            print(f"Early stopping at round {rnd} for {method_name}.")
            break

    global_encoder.load_state_dict(best_encoder)
    for h, state in zip(client_heads, best_heads):
        h.load_state_dict(state)

    val_probs = evaluate_personalised(global_encoder, client_heads, X_val, y_val, cfg)
    threshold, _ = find_best_threshold(y_val, val_probs)
    test_probs = evaluate_personalised(global_encoder, client_heads, X_test, y_test, cfg)
    test_metrics = evaluate_probs(y_test, test_probs, threshold)
    eps_rdp, best_order = accountant.epsilon()

    out = {
        "seed": seed,
        "method": method_name,
        "threshold": threshold,
        "rdp_epsilon": eps_rdp,
        "rdp_best_order": best_order,
        "delta": cfg.delta,
        **test_metrics,
    }
    return out, pd.DataFrame(history)


# ============================================================
# 9. Experiment runner
# ============================================================

def summarize_results(results_df: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1", "auprc", "auroc", "rdp_epsilon"]
    rows = []
    for method, g in results_df.groupby("method"):
        row = {"method": method, "n_seeds": len(g)}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().values
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_ci95_low"] = np.nan
                row[f"{m}_ci95_high"] = np.nan
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_low"] = mean - half
            row[f"{m}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auprc_mean", ascending=False)


def main(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    seed_everything(42)
    X, y = load_diabetes_csv(cfg.csv_path, cfg.target_col, cfg.positive_label)
    print(f"Loaded dataset: X={X.shape}, positives={int(y.sum())}, negatives={int(len(y) - y.sum())}")

    methods = [
        "NO_DP",
        "FIXED_HDP",
        "A1_ClipOnly",
        "A2_Clip+Budget",
        "A3_Clip+Budget+ClientSens",
        "A4_Full_PR_AS_HDP",
    ]

    all_results = []
    all_histories = []

    for seed in cfg.seeds:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = split_data(X, y, cfg, seed)
        print("\n" + "#" * 80)
        print(f"SEED = {seed}")
        print("#" * 80)

        for method in methods:
            print("\n" + "=" * 80)
            print(f"Running {method}")
            print("=" * 80)
            result, history = run_method(
                method,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                cfg,
                seed,
            )
            all_results.append(result)
            all_histories.append(history.assign(seed=seed))

    results_df = pd.DataFrame(all_results)
    history_df = pd.concat(all_histories, ignore_index=True)
    summary_df = summarize_results(results_df)

    results_path = os.path.join(cfg.output_dir, "per_seed_results.csv")
    history_path = os.path.join(cfg.output_dir, "round_history.csv")
    summary_path = os.path.join(cfg.output_dir, "summary_mean_std_ci.csv")
    config_path = os.path.join(cfg.output_dir, "config.json")

    results_df.to_csv(results_path, index=False)
    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nPER-SEED RESULTS")
    display(results_df)
    print("\nSUMMARY")
    display(summary_df)
    print("\nSaved outputs to:", cfg.output_dir)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for method in methods:
            g = history_df[history_df["method"] == method]
            curve = g.groupby("round")["val_auprc"].mean()
            plt.plot(curve.index, curve.values, label=method)
        plt.xlabel("Communication Round")
        plt.ylabel("Validation AUPRC")
        plt.title("Validation AUPRC Across Communication Rounds")
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception as e:
        print("Plotting failed:", e)


main(CFG)
