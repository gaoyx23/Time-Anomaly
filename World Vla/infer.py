"""
推理脚本：加载预训练的WorldVLA模型，在测试集上评估性能
"""
import argparse
import gzip
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============ 从vla.py复制的必要组件 ============

def read_csv_auto(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
	if file_path.endswith(".gz"):
		with gzip.open(file_path, "rt", encoding="utf-8") as f:
			return pd.read_csv(f, nrows=nrows)
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
		raise FileNotFoundError(f"Could not find train/test csv files in {version_dir}.")
	return train_files, test_files


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
		
		self.encoder = nn.LSTM(
			input_size=state_dim + action_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		
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
		
		self.policy_head = nn.Sequential(
			nn.Linear(hidden_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, action_dim),
		)

	def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x = torch.cat([states, actions], dim=-1)
		h, _ = self.encoder(x)
		
		z = self.to_latent(h[:, :-1, :])
		state_pred = self.state_decoder(z)
		action_pred = self.policy_head(h[:, :-1, :])
		
		return state_pred, action_pred

	def errors(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		state_pred, action_pred = self.forward(states, actions)
		
		state_target = states[:, 1:, :]
		action_target = actions[:, 1:, :]
		
		state_err = torch.mean((state_pred - state_target) ** 2, dim=(1, 2))
		action_err = torch.mean((action_pred - action_target) ** 2, dim=(1, 2))
		
		return state_err, action_err


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
	combined_scores = 0.7 * state_scores + 0.3 * action_scores
	
	return state_scores, action_scores, combined_scores


# ============ 推理函数 ============

def infer(args: argparse.Namespace) -> None:
	"""
	加载预训练模型并在测试集上评估
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	print(f"[Device] {device}")
	
	# 加载checkpoint
	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
	
	print(f"[Load] Loading checkpoint from {args.checkpoint}")
	ckpt = torch.load(args.checkpoint, map_location=device)
	
	state_cols = ckpt["state_cols"]
	action_cols = ckpt["action_cols"]
	window_size = ckpt["window"]
	stride = ckpt["stride"]
	threshold = ckpt["threshold"]
	
	print(f"[Config] state_dim={len(state_cols)}, action_dim={len(action_cols)}")
	print(f"[Config] window_size={window_size}, stride={stride}")
	print(f"[Config] threshold={threshold:.6f}")
	
	# 加载测试数据
	print(f"\n[Data] Loading test data from {args.data_dir}")
	train_files, test_files = find_split_files(args.data_dir)
	
	# 从完整训练集计算统计（保证与训练时一致）
	print(f"[Data] Computing statistics from training data...")
	train_states_all = []
	train_actions_all = []
	for fp in train_files:
		df = read_csv_auto(fp, nrows=args.max_train_rows)
		train_states_all.append(df[state_cols].to_numpy(dtype=np.float32))
		train_actions_all.append(df[action_cols].to_numpy(dtype=np.float32))
	
	train_states_arr = np.concatenate(train_states_all, axis=0)
	train_actions_arr = np.concatenate(train_actions_all, axis=0)
	
	state_mean = train_states_arr.mean(axis=0, keepdims=True)
	state_std = train_states_arr.std(axis=0, keepdims=True) + 1e-6
	action_mean = train_actions_arr.mean(axis=0, keepdims=True)
	action_std = train_actions_arr.std(axis=0, keepdims=True) + 1e-6
	
	# 加载完整测试数据
	test_states = []
	test_actions = []
	test_labels = []
	
	for fp in test_files:
		df = read_csv_auto(fp, nrows=args.max_test_rows)
		test_states.append(df[state_cols].to_numpy(dtype=np.float32))
		test_actions.append(df[action_cols].to_numpy(dtype=np.float32))
		
		# 自动检测标签列
		label_candidates = [c for c in df.columns if c.lower().startswith("attack")]
		if label_candidates:
			test_labels.append(df[label_candidates[0]].to_numpy(dtype=np.int64))
		else:
			print(f"[Warn] No attack label column in {fp}")
			test_labels.append(np.zeros(len(df), dtype=np.int64))
	
	test_states_arr = np.concatenate(test_states, axis=0)
	test_actions_arr = np.concatenate(test_actions, axis=0)
	test_label_arr = np.concatenate(test_labels, axis=0)
	
	# 归一化
	test_states_arr = (test_states_arr - state_mean) / state_std
	test_actions_arr = (test_actions_arr - action_mean) / action_std
	
	# 构建窗口
	test_state_windows = build_windows(test_states_arr, window_size=window_size, stride=stride)
	test_action_windows = build_windows(test_actions_arr, window_size=window_size, stride=stride)
	test_window_labels = build_label_windows(test_label_arr, window_size=window_size, stride=stride)
	
	print(f"[Data] test windows: {test_state_windows.shape}")
	print(f"[Data] anomaly ratio: {test_window_labels.mean():.4f}")
	
	# 重构模型
	model = WorldVLA(
		state_dim=len(state_cols),
		action_dim=len(action_cols),
		hidden_dim=128,
		latent_dim=64,
		num_layers=2,
		dropout=0.1,
	)
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	
	# 推理
	print(f"\n[Infer] Computing anomaly scores...")
	test_state_scores, test_action_scores, test_combined = collect_scores(
		model, test_state_windows, test_action_windows, args.batch, device
	)
	
	# 评估
	y_pred = (test_combined > threshold).astype(np.int64)
	metrics = binary_metrics(test_window_labels, y_pred)
	
	print("\n" + "=" * 50)
	print("EVALUATION RESULTS")
	print("=" * 50)
	print(f"Threshold:  {threshold:.6f}")
	print(f"Precision:  {metrics['precision']:.4f}")
	print(f"Recall:     {metrics['recall']:.4f}")
	print(f"F1-Score:   {metrics['f1']:.4f}")
	print(f"TP={metrics['tp']:5d} FP={metrics['fp']:5d} FN={metrics['fn']:5d} TN={metrics['tn']:5d}")
	
	print("\n[Score Stats]")
	print(f"State Error:    mean={test_state_scores.mean():.6f} std={test_state_scores.std():.6f}")
	print(f"Action Error:   mean={test_action_scores.mean():.6f} std={test_action_scores.std():.6f}")
	print(f"Combined Score: mean={test_combined.mean():.6f} std={test_combined.std():.6f}")
	print(f"  Min={test_combined.min():.6f} Max={test_combined.max():.6f}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Evaluate pretrained WorldVLA model on HAI test set"
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=r"c:\Users\gyx\Desktop\RL4TIME\World Vla\worldvla_hai.pt",
		help="Path to saved model checkpoint",
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default=r"c:\Users\gyx\Desktop\RL4TIME\hai\hai-21.03",
		help="Path to HAI version directory",
	)
	parser.add_argument("--batch", type=int, default=256)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument(
		"--max-test-rows",
		type=int,
		default=None,
		help="Debug: limit rows per test file",
	)
	parser.add_argument(
		"--max-train-rows",
		type=int,
		default=None,
		help="Debug: limit rows per train file",
	)
	return parser


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	infer(args)
