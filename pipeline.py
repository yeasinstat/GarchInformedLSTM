"""
GARCH-Informed LSTM (GINN) — Web App core pipeline
====================================================
Ported from the user's GINN script. Pipeline:
  1. y_t -> r_t = (y_t - y_{t-1}) / y_{t-1}                [returns]
  2. AR(p) on r_t -> mu_hat_t                                [mean forecast]
     GARCH(p,q) on r_t -> sigma2_GARCH_t                     [variance forecast]
  3. Ground truth variance: sigma2_t = (r_t - mu_hat_t)^2
  4. LSTM input = past sigma2_t sequences (NOT y_t)
     LSTM output = sigma2_hat_t_LSTM
  5. Loss = (1-lambda) * MSE(sigma2_t, sigma2_hat_LSTM)
          + lambda * MSE(sigma2_hat_GARCH, sigma2_hat_LSTM)
     lambda=0.0 -> Standard LSTM (pure ground truth)
     lambda=1.0 -> GINN-0 (pure GARCH matching)
"""
import warnings, time, copy
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model

np.random.seed(42)
torch.manual_seed(42)


def format_time(s):
    if s < 60:
        return f"{s:.0f}s"
    return f"{int(s // 60)}m {s % 60:.0f}s"


# ═══════════════════════════════════════════════════════════
# Returns, AR, GARCH
# ═══════════════════════════════════════════════════════════

def compute_returns(data):
    return np.diff(data) / data[:-1]


def ar_autotune(train_returns):
    max_lag = max(1, min(5, len(train_returns) // 5))
    best_aic, best_lag, best_model = float("inf"), None, None
    aic_table = []
    for lag in range(1, max_lag + 1):
        try:
            mdl = AutoReg(train_returns, lags=lag).fit()
            aic_table.append(dict(lag=lag, aic=float(mdl.aic), ok=True))
            if mdl.aic < best_aic:
                best_aic, best_lag, best_model = mdl.aic, lag, mdl
        except Exception:
            aic_table.append(dict(lag=lag, aic=None, ok=False))
    return best_lag, best_model, aic_table


def garch_autotune(train_returns, ar_lag, progress_fn=None):
    combos = [(p, q, mean, dist)
              for p in [1, 2, 3] for q in [1, 2, 3]
              for mean in ["Zero", "AR"] for dist in ["normal", "t"]]
    best_aic, best_cfg, best_fit = float("inf"), None, None
    aic_table = []
    total = len(combos)
    for i, (p, q, mean, dist) in enumerate(combos):
        if progress_fn:
            progress_fn(i / total, desc=f"GARCH auto-tune: combo {i + 1}/{total} "
                                        f"(p={p}, q={q}, mean={mean}, dist={dist})")
        try:
            mdl = arch_model(train_returns * 100, vol="Garch", p=p, q=q, mean=mean,
                              lags=ar_lag if mean == "AR" else None, dist=dist, rescale=False)
            fit = mdl.fit(disp="off")
            aic_table.append(dict(p=p, q=q, mean=mean, dist=dist, aic=float(fit.aic), ok=True))
            if fit.aic < best_aic:
                best_aic, best_cfg, best_fit = fit.aic, (p, q, mean, dist), fit
        except Exception:
            aic_table.append(dict(p=p, q=q, mean=mean, dist=dist, aic=None, ok=False))
    if progress_fn:
        progress_fn(1.0, desc=f"GARCH auto-tune: {total}/{total} combos complete")
    return best_cfg, best_fit, aic_table


def fill_leading_nan(arr):
    """GARCH with mean='AR' leaves the first AR_LAG entries of conditional_volatility
    as NaN (burn-in — not enough history yet for the AR mean equation). Fill them
    with the first valid value, matching how the AR mean forecast's own burn-in
    is already handled."""
    arr = np.asarray(arr, dtype=float)
    if not np.isnan(arr).any():
        return arr
    valid = arr[~np.isnan(arr)]
    fill_val = valid[0] if len(valid) else 0.0
    return np.where(np.isnan(arr), fill_val, arr)


def prepare_sequences(state, seq_len):
    """Builds scaled sigma^2 sequences + train/val/test tensors + aligned GARCH
    guidance tensor, shared by both the LSTM-tuning step and the final GINN run."""
    sigma2_gt_train = state["sigma2_gt_train"]; sigma2_gt_test = state["sigma2_gt_test"]
    sigma2_garch_train = state["sigma2_garch_train"]; sigma2_garch_test = state["sigma2_garch_test"]
    split = state["split"]

    scaler = MinMaxScaler()
    sigma2_gt_all = np.concatenate([sigma2_gt_train, sigma2_gt_test])
    sigma2_scaled = scaler.fit_transform(sigma2_gt_all.reshape(-1, 1)).flatten()
    sigma2_garch_all = np.concatenate([sigma2_garch_train, sigma2_garch_test])
    sigma2_garch_scaled = scaler.transform(sigma2_garch_all.reshape(-1, 1)).flatten()

    X_all, y_all = make_seq(sigma2_scaled, seq_len)
    seq_split = split - seq_len
    if seq_split < 5 or len(X_all) - seq_split < 3:
        raise ValueError(
            f"Sequence length ({seq_len}) is too large for this dataset (only {len(sigma2_gt_all)} "
            f"variance points after the returns transform). Reduce the sequence length or use more data.")

    X_train, y_train = X_all[:seq_split], y_all[:seq_split]
    X_test, y_test = X_all[seq_split:], y_all[seq_split:]
    n_test = len(y_test)

    garch_tr_scaled = sigma2_garch_scaled[seq_len:seq_len + len(y_train)]
    garch_te_scaled = sigma2_garch_scaled[split:split + n_test]

    val_split = int(len(X_train) * 0.8)
    if val_split < 2 or len(X_train) - val_split < 2:
        val_split = max(2, len(X_train) - 2)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    X_tr_t = torch.FloatTensor(X_tr).unsqueeze(-1); y_tr_t = torch.FloatTensor(y_tr).unsqueeze(-1)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(-1); y_val_t = torch.FloatTensor(y_val).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1); y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)
    X_train_full_t = torch.FloatTensor(X_train).unsqueeze(-1); y_train_full_t = torch.FloatTensor(y_train).unsqueeze(-1)
    garch_tr_t = torch.FloatTensor(garch_tr_scaled[:val_split]).unsqueeze(-1)

    bs = min(16, len(X_tr))
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)

    return dict(scaler=scaler, X_tr_t=X_tr_t, y_tr_t=y_tr_t, X_val_t=X_val_t, y_val_t=y_val_t,
                X_test_t=X_test_t, y_test_t=y_test_t, X_train_full_t=X_train_full_t,
                y_train_full_t=y_train_full_t, garch_tr_t=garch_tr_t, train_loader=train_loader,
                garch_te_scaled=garch_te_scaled, n_test=n_test, val_split=val_split,
                len_y_train=len(y_train))


def make_seq(s, q):
    X, y = [], []
    for i in range(len(s) - q):
        X.append(s[i:i + q])
        y.append(s[i + q])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# LSTM model + GINN loss
# ═══════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        dropout = float(dropout)
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GINNLoss(nn.Module):
    """Loss = (1-lambda) * MSE(pred, ground_truth) + lambda * MSE(pred, garch)
    lambda=0.0 -> Standard LSTM (pure ground truth)
    lambda=1.0 -> GINN-0 (pure GARCH-matching)"""
    def __init__(self, lambda_weight=0.5):
        super().__init__()
        self.lam = lambda_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, gt, garch):
        loss_gt = self.mse(pred, gt)
        loss_garch = self.mse(pred, garch)
        total = (1 - self.lam) * loss_gt + self.lam * loss_garch
        return total, loss_gt, loss_garch


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def evaluate_variance(model, Xt, yt, scaler):
    model.eval()
    with torch.no_grad():
        ps = model(Xt).squeeze().numpy()
    pr = scaler.inverse_transform(ps.reshape(-1, 1)).flatten()
    ac = scaler.inverse_transform(yt.squeeze().numpy().reshape(-1, 1)).flatten()
    rmse = float(np.sqrt(mean_squared_error(ac, pr)))
    mae = float(mean_absolute_error(ac, pr))
    denom = np.abs(ac) + np.abs(pr)
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(ac - pr) / np.where(denom == 0, 1, denom))) * 100)
    return pr, ac, rmse, mae, smape


def train_ginn(hp, train_loader_gt_only, X_val, y_val, X_test, y_test, scaler,
                garch_tr_t, lam, epochs=500, patience=30, batch_size=16,
                X_tr=None, y_tr=None):
    """Trains one LSTM with GINNLoss(lambda=lam).
    If lam == 0.0, trains purely on (X,y) pairs (Standard LSTM, no GARCH tensor needed).
    Otherwise builds a (X,y,garch) loader for the informed loss."""
    model = LSTMModel(hp["hidden"], hp["layers"], hp["dropout"])
    crit = GINNLoss(lam)
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    mse_only = nn.MSELoss()
    best_val, counter, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    hist = {"total": [], "gt": [], "garch": [], "val": []}

    if lam == 0.0:
        loader = train_loader_gt_only
        for epoch in range(epochs):
            model.train()
            et = eg = 0.0
            for bX, by in loader:
                optim.zero_grad()
                loss, lg, _ = crit(model(bX), by, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                et += loss.item(); eg += lg.item()
            n = len(loader)
            hist["total"].append(et / n); hist["gt"].append(eg / n); hist["garch"].append(0.0)
            model.eval()
            with torch.no_grad():
                vl = mse_only(model(X_val), y_val).item()
            hist["val"].append(vl)
            if vl < best_val:
                best_val, counter, best_state = vl, 0, copy.deepcopy(model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
    else:
        bs = min(batch_size, len(X_tr))
        ds = TensorDataset(X_tr, y_tr, garch_tr_t)
        loader = DataLoader(ds, batch_size=bs, shuffle=False)
        for epoch in range(epochs):
            model.train()
            et = eg = eG = 0.0
            for bX, by, bg in loader:
                optim.zero_grad()
                loss, lg, lG = crit(model(bX), by, bg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                et += loss.item(); eg += lg.item(); eG += lG.item()
            n = len(loader)
            hist["total"].append(et / n); hist["gt"].append(eg / n); hist["garch"].append(eG / n)
            model.eval()
            with torch.no_grad():
                vl = mse_only(model(X_val), y_val).item()
            hist["val"].append(vl)
            if vl < best_val:
                best_val, counter, best_state = vl, 0, copy.deepcopy(model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    pr, ac, rmse, mae, smape = evaluate_variance(model, X_test, y_test, scaler)
    return model, pr, ac, rmse, mae, smape, hist, count_params(model)


def grid_search_lstm(param_grid, train_loader, Xv, yv, scaler, tune_epochs=50, progress_fn=None):
    results = []
    combos = [(h, l, lr, d)
              for h in param_grid["hidden_size"]
              for l in param_grid["num_layers"]
              for lr in param_grid["lr"]
              for d in param_grid["dropout"]]
    total = len(combos)
    start_t = time.time()
    for i, (h, l, lr, d) in enumerate(combos):
        if progress_fn:
            elapsed = time.time() - start_t
            avg = elapsed / i if i > 0 else 0.0
            eta = avg * (total - i)
            eta_str = f", ETA {format_time(eta)}" if i > 0 else ""
            progress_fn(i / total, desc=f"Grid search: combo {i + 1}/{total} "
                                        f"(hidden={int(h)}, layers={int(l)}, lr={lr:.4f}, dropout={d:.2f}) "
                                        f"— elapsed {format_time(elapsed)}{eta_str}")
        m = LSTMModel(int(h), int(l), float(d))
        opt = torch.optim.Adam(m.parameters(), lr=float(lr))
        crit = GINNLoss(0.0)  # pure ground-truth loss for tuning
        m.train()
        for _ in range(tune_epochs):
            for bX, by in train_loader:
                opt.zero_grad()
                loss, _, _ = crit(m(bX), by, by)
                loss.backward()
                opt.step()
        _, _, rv, _, _ = evaluate_variance(m, Xv, yv, scaler)
        results.append(dict(hidden=int(h), layers=int(l), lr=float(lr), dropout=float(d), rmse=rv))
    if progress_fn:
        progress_fn(1.0, desc=f"Grid search: {total}/{total} combos complete "
                              f"in {format_time(time.time() - start_t)}")
    return sorted(results, key=lambda x: x["rmse"])[0], results


def bayes_search_lstm(train_loader, Xv, yv, scaler, hidden_range, layers_range, lr_range,
                       dropout_range, n_calls=15, tune_epochs=50, random_state=42, progress_fn=None):
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    specs, fixed = [], {}

    def add_dim(name, low, high, is_int):
        low = float(low); high = float(high)
        if low > high:
            low, high = high, low
        if is_int:
            low, high = int(round(low)), int(round(high))
        if low == high:
            fixed[name] = low
        else:
            specs.append((name, low, high, is_int))

    add_dim("hidden", hidden_range[0], hidden_range[1], True)
    add_dim("layers", layers_range[0], layers_range[1], True)
    add_dim("lr", lr_range[0], lr_range[1], False)
    add_dim("dropout", dropout_range[0], dropout_range[1], False)

    def _train_and_eval(hidden, layers, lr, dropout):
        m = LSTMModel(int(round(hidden)), int(round(layers)), float(dropout))
        opt = torch.optim.Adam(m.parameters(), lr=float(lr))
        crit = GINNLoss(0.0)
        m.train()
        for _ in range(tune_epochs):
            for bX, by in train_loader:
                opt.zero_grad()
                loss, _, _ = crit(m(bX), by, by)
                loss.backward()
                opt.step()
        _, _, rv, _, _ = evaluate_variance(m, Xv, yv, scaler)
        return rv

    if not specs:
        if progress_fn:
            progress_fn(0.0, desc="Evaluating fixed hyperparameters...")
        rv = _train_and_eval(fixed["hidden"], fixed["layers"], fixed["lr"], fixed["dropout"])
        if progress_fn:
            progress_fn(1.0, desc="Done")
        best = dict(hidden=int(fixed["hidden"]), layers=int(fixed["layers"]),
                    lr=float(fixed["lr"]), dropout=float(fixed["dropout"]), rmse=rv)
        return best, [best]

    dimensions = [Integer(low, high, name=name) if is_int else Real(low, high, name=name)
                  for name, low, high, is_int in specs]

    @use_named_args(dimensions)
    def objective(**params):
        p = {**fixed, **params}
        return _train_and_eval(p["hidden"], p["layers"], p["lr"], p["dropout"])

    n_calls = max(int(n_calls), len(dimensions) + 2)
    n_initial = max(2, min(5, n_calls // 2))
    bstart = time.time()

    def _cb(res):
        if progress_fn:
            done = len(res.x_iters)
            elapsed = time.time() - bstart
            avg = elapsed / done if done > 0 else 0.0
            eta = avg * (n_calls - done)
            eta_str = f", ETA {format_time(eta)}" if done < n_calls else ""
            progress_fn(done / n_calls, desc=f"Bayesian optimization: evaluation {done}/{n_calls} "
                                              f"(best RMSE so far: {min(res.func_vals):.6f}) "
                                              f"— elapsed {format_time(elapsed)}{eta_str}")

    result = gp_minimize(objective, dimensions, n_calls=n_calls, n_initial_points=n_initial,
                          random_state=random_state, verbose=False,
                          callback=_cb if progress_fn else None)

    def _to_hp(x_vals, rmse):
        params = {name: v for (name, *_), v in zip(specs, x_vals)}
        params = {**fixed, **params}
        return dict(hidden=int(round(params["hidden"])), layers=int(round(params["layers"])),
                    lr=float(params["lr"]), dropout=float(params["dropout"]), rmse=float(rmse))

    best = _to_hp(result.x, result.fun)
    all_results = [_to_hp(x, y) for x, y in zip(result.x_iters, result.func_vals)]
    return best, all_results


def build_grid_values(mn, mx, step, is_int):
    mn, mx, step = float(mn), float(mx), float(step)
    if mx < mn:
        mn, mx = mx, mn
    if step <= 0:
        step = 1.0 if is_int else 0.05
    vals = np.arange(mn, mx + step / 2, step)
    if is_int:
        vals = np.unique(np.round(vals).astype(int))
    else:
        vals = np.unique(np.round(vals, 6))
    return vals.tolist()