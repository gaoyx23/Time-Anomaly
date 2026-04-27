import argparse
import gzip
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def is_lfs_pointer(file_path: str) -> bool:
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			head = f.readline().strip()
		return head.startswith("version https://git-lfs.github.com/spec/v1")
	except UnicodeDecodeError:
		return False


def read_csv_auto(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
	if file_path.endswith(".gz"):
		with gzip.open(file_path, "rt", encoding="utf-8") as f:
			return pd.read_csv(f, nrows=nrows)

	if is_lfs_pointer(file_path):
		raise RuntimeError(
			f"File {file_path} is a Git LFS pointer, not real data. "
			"Please run 'git lfs pull' in the hai directory first."
		)

	return pd.read_csv(file_path, nrows=nrows)


def find_split_files(version_dir: str) -> Tuple[List[str], List[str]]:
	train_files: List[str] = []
	test_files: List[str] = []

	for name in sorted(os.listdir(version_dir)):
		lower = name.lower()
		if not (lower.endswith(".csv") or lower.endswith(".csv.gz")):
			continue
		full = os.path.join(version_dir, name)
		if lower.startswith("train"):
			train_files.append(full)
		elif lower.startswith("test"):
			test_files.append(full)

	if not train_files or not test_files:
		raise FileNotFoundError(
			f"Could not find train/test csv files in {version_dir}."
		)
	return train_files, test_files


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, List[str], List[str]]:
	cols = list(df.columns)
	time_col = "timestamp" if "timestamp" in cols else "time" if "time" in cols else cols[0]

	attack_candidates = ["attack", "Attack"]
	label_col = None
	for c in attack_candidates:
		if c in cols:
			label_col = c
			break
	if label_col is None:
		raise ValueError("Could not find attack label column (attack/Attack).")

	label_cols = [c for c in cols if c.lower().startswith("attack")]
	
	# Split features into actions (setpoints, *D suffix) and state (observations)
	all_features = [c for c in cols if c not in label_cols and c != time_col]
	action_cols = [c for c in all_features if c.endswith("D")]
	state_cols = [c for c in all_features if c not in action_cols]
	
	return time_col, label_col, state_cols, action_cols


def build_windows(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
	if len(arr) < window_size:
		return np.zeros((0, window_size, arr.shape[1]), dtype=np.float32)

	windows = []
	for i in range(0, len(arr) - window_size + 1, stride):
		windows.append(arr[i : i + window_size])
	return np.asarray(windows, dtype=np.float32)


def build_label_windows(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
	if len(labels) < window_size:
		return np.zeros((0,), dtype=np.int64)

	y = []
	for i in range(0, len(labels) - window_size + 1, stride):
		# Window-level anomaly label: any anomaly inside window.
		y.append(int(labels[i : i + window_size].max() > 0))
	return np.asarray(y, dtype=np.int64)


class SequenceDataset(Dataset):
	def __init__(self, states: np.ndarray, actions: np.ndarray):
		self.states = torch.from_numpy(states).float()
		self.actions = torch.from_numpy(actions).float()

	def __len__(self) -> int:
		return self.states.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.states[idx], self.actions[idx]


class WorldVLA(nn.Module):
	"""
	WorldVLA: Action-conditional world model with policy head for ICS anomaly detection.
	
	Inputs per timestep:
	  - state_t: sensor observations (excluding setpoints)
	  - action_t: setpoint commands (control signals)
	
	Outputs:
	  - Predicts state_{t+1} given (state_t, action_t)
	  - Also predicts "natural" action from state_t (policy head)
	
	Anomaly signals:
	  - Reconstruction error: ||state_{t+1}^pred - state_{t+1}^actual||
	  - Action deviation: how much actual action deviates from policy prediction
	"""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		hidden_dim: int = 128,
		latent_dim: int = 64,
		num_layers: int = 2,
		dropout: float = 0.1,
	):
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		
		# Joint LSTM encoder on state+action concatenation
		self.encoder = nn.LSTM(
			input_size=state_dim + action_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		
		# Next-state decoder (state prediction)
		self.to_latent = nn.Sequential(
			nn.Linear(hidden_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
		)
		self.state_decoder = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, state_dim),
		)
		
		# Policy head: state -> natural action (what action should be taken given state)
		self.policy_head = nn.Sequential(
			nn.Linear(hidden_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, action_dim),
		)

	def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			states: (batch, seq_len, state_dim)
			actions: (batch, seq_len, action_dim)
		
		Returns:
			state_pred: (batch, seq_len-1, state_dim)
			action_pred: (batch, seq_len-1, action_dim)
		"""
		# Concatenate state and action
		x = torch.cat([states, actions], dim=-1)  # (batch, seq_len, state_dim+action_dim)
		
		# Encode
		h, _ = self.encoder(x)  # (batch, seq_len, hidden_dim)
		
		# Decode states (t to t+1)
		z = self.to_latent(h[:, :-1, :])  # (batch, seq_len-1, latent_dim)
		state_pred = self.state_decoder(z)  # (batch, seq_len-1, state_dim)
		
		# Predict natural actions from hidden state
		action_pred = self.policy_head(h[:, :-1, :])  # (batch, seq_len-1, action_dim)
		
		return state_pred, action_pred

	def errors(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute reconstruction and action errors per window.
		
		Returns:
			state_error: (batch,) MSE for state reconstruction
			action_error: (batch,) MSE for action prediction
		"""
		state_pred, action_pred = self.forward(states, actions)
		
		state_target = states[:, 1:, :]  # (batch, seq_len-1, state_dim)
		action_target = actions[:, 1:, :]  # (batch, seq_len-1, action_dim)
		
		state_err = torch.mean((state_pred - state_target) ** 2, dim=(1, 2))  # (batch,)
		action_err = torch.mean((action_pred - action_target) ** 2, dim=(1, 2))  # (batch,)
		
		return state_err, action_err


@dataclass
class PreparedData:
	train_states: np.ndarray
	train_actions: np.ndarray
	test_states: np.ndarray
	test_actions: np.ndarray
	test_window_labels: np.ndarray
	state_cols: List[str]
	action_cols: List[str]


def prepare_hai_data(
	version_dir: str,
	window_size: int,
	stride: int,
	max_train_rows_per_file: Optional[int] = None,
	max_test_rows_per_file: Optional[int] = None,
) -> PreparedData:
	train_files, test_files = find_split_files(version_dir)

	# Infer schema from first train file.
	schema_df = read_csv_auto(train_files[0], nrows=5)
	_, label_col, state_cols, action_cols = detect_columns(schema_df)

	train_states = []
	train_actions = []
	for fp in train_files:
		df = read_csv_auto(fp, nrows=max_train_rows_per_file)
		train_states.append(df[state_cols].to_numpy(dtype=np.float32))
		train_actions.append(df[action_cols].to_numpy(dtype=np.float32))
	
	train_states_arr = np.concatenate(train_states, axis=0)
	train_actions_arr = np.concatenate(train_actions, axis=0)

	# Normalize with train statistics only.
	state_mean = train_states_arr.mean(axis=0, keepdims=True)
	state_std = train_states_arr.std(axis=0, keepdims=True) + 1e-6
	action_mean = train_actions_arr.mean(axis=0, keepdims=True)
	action_std = train_actions_arr.std(axis=0, keepdims=True) + 1e-6
	
	train_states_arr = (train_states_arr - state_mean) / state_std
	train_actions_arr = (train_actions_arr - action_mean) / action_std

	test_states = []
	test_actions = []
	test_labels = []
	for fp in test_files:
		df = read_csv_auto(fp, nrows=max_test_rows_per_file)
		test_states.append(df[state_cols].to_numpy(dtype=np.float32))
		test_actions.append(df[action_cols].to_numpy(dtype=np.float32))
		test_labels.append(df[label_col].to_numpy(dtype=np.int64))

	test_states_arr = np.concatenate(test_states, axis=0)
	test_actions_arr = np.concatenate(test_actions, axis=0)
	test_label_arr = np.concatenate(test_labels, axis=0)
	
	test_states_arr = (test_states_arr - state_mean) / state_std
	test_actions_arr = (test_actions_arr - action_mean) / action_std

	train_state_windows = build_windows(train_states_arr, window_size=window_size, stride=stride)
	train_action_windows = build_windows(train_actions_arr, window_size=window_size, stride=stride)
	
	test_state_windows = build_windows(test_states_arr, window_size=window_size, stride=stride)
	test_action_windows = build_windows(test_actions_arr, window_size=window_size, stride=stride)
	test_window_labels = build_label_windows(test_label_arr, window_size=window_size, stride=stride)

	return PreparedData(
		train_states=train_state_windows,
		train_actions=train_action_windows,
		test_states=test_state_windows,
		test_actions=test_action_windows,
		test_window_labels=test_window_labels,
		state_cols=state_cols,
		action_cols=action_cols,
	)


def train_worldvla(
	model: WorldVLA,
	train_states: np.ndarray,
	train_actions: np.ndarray,
	batch_size: int,
	epochs: int,
	lr: float,
	device: torch.device,
) -> None:
	ds = SequenceDataset(train_states, train_actions)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()
	model.to(device)

	print(f"[Train] device={device}, windows={len(ds)}, epochs={epochs}")
	for epoch in range(1, epochs + 1):
		model.train()
		running_state = 0.0
		running_action = 0.0
		steps = 0
		
		for states, actions in loader:
			states = states.to(device)
			actions = actions.to(device)
			
			state_pred, action_pred = model(states, actions)
			state_target = states[:, 1:, :]
			action_target = actions[:, 1:, :]
			
			state_loss = criterion(state_pred, state_target)
			action_loss = criterion(action_pred, action_target)
			loss = state_loss + 0.5 * action_loss  # Joint loss

			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			running_state += state_loss.item()
			running_action += action_loss.item()
			steps += 1

		if epoch % max(1, epochs // 10) == 0 or epoch == 1:
			print(f"[Train] epoch={epoch:03d}/{epochs} state_loss={running_state / max(1, steps):.6f} action_loss={running_action / max(1, steps):.6f}")


def collect_scores(
	model: WorldVLA,
	states: np.ndarray,
	actions: np.ndarray,
	batch_size: int,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	ds = SequenceDataset(states, actions)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

	model.eval()
	state_out = []
	action_out = []
	
	with torch.no_grad():
		for batch_states, batch_actions in loader:
			batch_states = batch_states.to(device)
			batch_actions = batch_actions.to(device)
			
			state_err, action_err = model.errors(batch_states, batch_actions)
			state_out.append(state_err.cpu().numpy())
			action_out.append(action_err.cpu().numpy())
	
	state_scores = np.concatenate(state_out, axis=0)
	action_scores = np.concatenate(action_out, axis=0)
	# Combined anomaly score
	combined_scores = 0.7 * state_scores + 0.3 * action_scores
	
	return state_scores, action_scores, combined_scores


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))

	precision = tp / (tp + fp + 1e-6)
	recall = tp / (tp + fn + 1e-6)
	f1 = 2 * precision * recall / (precision + recall + 1e-6)
	return {
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"tn": tn,
	}


def run(args: argparse.Namespace) -> None:
	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

	# RL note: HAI here is an offline logged detection task. Without an interactive
	# control loop and environment reward dynamics, RL usually does not improve
	# robustness over a strong reconstruction baseline.
	if args.use_rl:
		print("[Info] --use-rl was requested, but RL is disabled for offline HAI logs.")
		print("[Info] Falling back to pure WorldVLA reconstruction training.")

	data = prepare_hai_data(
		version_dir=args.data_dir,
		window_size=args.window,
		stride=args.stride,
		max_train_rows_per_file=args.max_train_rows,
		max_test_rows_per_file=args.max_test_rows,
	)

	print(f"[Data] state_dim={len(data.state_cols)}, action_dim={len(data.action_cols)}")
	print(f"[Data] train: states={data.train_states.shape}, actions={data.train_actions.shape}")
	print(f"[Data] test: states={data.test_states.shape}, actions={data.test_actions.shape}")
	print(f"[Data] test_anomaly_ratio={data.test_window_labels.mean():.4f}")
	print(f"[Data] State cols: {data.state_cols[:5]}... ({len(data.state_cols)} total)")
	print(f"[Data] Action cols: {data.action_cols}")

	model = WorldVLA(
		state_dim=len(data.state_cols),
		action_dim=len(data.action_cols),
		hidden_dim=args.hidden,
		latent_dim=args.latent,
		num_layers=args.layers,
		dropout=args.dropout,
	)

	train_worldvla(
		model=model,
		train_states=data.train_states,
		train_actions=data.train_actions,
		batch_size=args.batch,
		epochs=args.epochs,
		lr=args.lr,
		device=device,
	)

	train_state_scores, train_action_scores, train_combined = collect_scores(
		model, data.train_states, data.train_actions, args.batch, device
	)
	test_state_scores, test_action_scores, test_combined = collect_scores(
		model, data.test_states, data.test_actions, args.batch, device
	)

	threshold = float(np.percentile(train_combined, args.threshold_percentile))
	y_pred = (test_combined > threshold).astype(np.int64)

	metrics = binary_metrics(data.test_window_labels, y_pred)

	print("\n================= Evaluation =================")
	print(f"threshold_percentile={args.threshold_percentile}")
	print(f"threshold={threshold:.6f}")
	print(f"precision={metrics['precision']:.4f}")
	print(f"recall={metrics['recall']:.4f}")
	print(f"f1={metrics['f1']:.4f}")
	print(
		f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} tn={metrics['tn']}"
	)
	
	print("\n[Debug] Score statistics:")
	print(f"  Train combined: mean={train_combined.mean():.4f} std={train_combined.std():.4f}")
	print(f"  Test combined:  mean={test_combined.mean():.4f} std={test_combined.std():.4f}")
	print(f"  Test state:     mean={test_state_scores.mean():.4f} std={test_state_scores.std():.4f}")
	print(f"  Test action:    mean={test_action_scores.mean():.4f} std={test_action_scores.std():.4f}")

	if args.save_model:
		os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
		ckpt = {
			"state_dict": model.state_dict(),
			"state_cols": data.state_cols,
			"action_cols": data.action_cols,
			"window": args.window,
			"stride": args.stride,
			"threshold": threshold,
		}
		torch.save(ckpt, args.save_model)
		print(f"[Save] model checkpoint: {args.save_model}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="WorldVLA anomaly detection on HAI dynamic ICS datasets"
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default=r"c:\Users\gyx\Desktop\RL4TIME\hai\hai-21.03",
		help="Path to HAI version directory (contains train*.csv(.gz), test*.csv(.gz))",
	)
	parser.add_argument("--window", type=int, default=60)
	parser.add_argument("--stride", type=int, default=5)
	parser.add_argument("--hidden", type=int, default=128)
	parser.add_argument("--latent", type=int, default=64)
	parser.add_argument("--layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--batch", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--threshold-percentile", type=float, default=99.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--use-rl", action="store_true")
	parser.add_argument(
		"--save-model", type=str, default=r"c:\Users\gyx\Desktop\RL4TIME\World Vla\worldvla_hai.pt"
	)
	parser.add_argument(
		"--max-train-rows",
		type=int,
		default=None,
		help="Debug option: cap rows loaded from each train file",
	)
	parser.add_argument(
		"--max-test-rows",
		type=int,
		default=None,
		help="Debug option: cap rows loaded from each test file",
	)
	return parser


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	run(args)