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
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset
import requests
import json


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


# ===============================================
# 1. 世界模型 - 体检仪 & 手术模拟舱
# ===============================================
class HealthcheckSystem(nn.Module):
	"""
	世界模型 - 承担两重角色：
	1. 体检仪：通过预测下一步状态，对比真实值产生预测误差 -> 异常检测
	2. 手术模拟舱：在虚拟环境推演，为RL提供梦境训练空间
	"""
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		hidden_dim: int = 128,
		latent_dim: int = 64,
		num_layers: int = 2,
		dropout: float = 0.1,
		action_min: Optional[np.ndarray] = None,
		action_max: Optional[np.ndarray] = None,
	):
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		
		# 【新增】记录训练数据的动作范围，用于detection out-of-distribution动作
		if action_min is not None and action_max is not None:
			self.action_min = torch.tensor(action_min, dtype=torch.float32)
			self.action_max = torch.tensor(action_max, dtype=torch.float32)
		else:
			self.action_min = None
			self.action_max = None
		
		# LSTM编码器：捕捉时序依赖
		self.encoder = nn.LSTM(
			input_size=state_dim + action_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		
		# 潜在表示
		self.to_latent = nn.Sequential(
			nn.Linear(hidden_dim, latent_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 2),  # 【改进】增加Dropout防止过拟合
			nn.Linear(latent_dim, latent_dim),
		)
		
		# 下一状态解码器
		self.state_decoder = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 2),  # 【改进】增加Dropout
			nn.Linear(hidden_dim, state_dim),
		)
		
		# 策略头：预测"自然"动作（用于异常检测）
		self.policy_head = nn.Sequential(
			nn.Linear(hidden_dim, latent_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 2),  # 【改进】增加Dropout
			nn.Linear(latent_dim, action_dim),
		)

	def forward(
		self, 
		states: torch.Tensor, 
		actions: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		前向传播
		
		Args:
			states: (batch, seq_len, state_dim)
			actions: (batch, seq_len, action_dim)
		
		Returns:
			state_pred: 预测下一状态 (batch, seq_len-1, state_dim)
			action_pred: 预测下一动作 (batch, seq_len-1, action_dim)  
			state_error: 【修改】重构误差作为异常指标 (batch, seq_len-1, 1)
		"""
		x = torch.cat([states, actions], dim=-1)
		h, _ = self.encoder(x)
		
		# 预测下一状态
		z = self.to_latent(h[:, :-1, :])
		state_pred = self.state_decoder(z)
		
		# 预测自然动作
		action_pred = self.policy_head(h[:, :-1, :])
		
		# 【修改】异常指标 = 重构误差（MSE of predicted vs actual next state）
		state_target = states[:, 1:, :]  # 真实下一状态
		state_error = torch.mean((state_pred - state_target) ** 2, dim=2, keepdim=True)  # (batch, seq_len-1, 1)
		
		return state_pred, action_pred, state_error

	def health_check_errors(
		self, 
		states: torch.Tensor, 
		actions: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		体检仪模式：计算预测误差（检查指标）
		
		Returns:
			state_error: (batch,) 状态重建误差
			action_error: (batch,) 动作预测误差
		"""
		state_pred, action_pred, _ = self.forward(states, actions)
		
		state_target = states[:, 1:, :]
		action_target = actions[:, 1:, :]
		
		state_err = torch.mean((state_pred - state_target) ** 2, dim=(1, 2))
		action_err = torch.mean((action_pred - action_target) ** 2, dim=(1, 2))
		
		return state_err, action_err

	def simulate_step(
		self, 
		state: torch.Tensor, 
		action: torch.Tensor,
		device: Optional[torch.device] = None
	) -> torch.Tensor:
		"""
		手术模拟舱模式：推演单步
		用于RL的梦境环境模拟
		
		Args:
			state: (state_dim,) 当前状态
			action: (action_dim,) 执行的动作
			
		Returns:
			next_state: (state_dim,) 预测的下一状态
		"""
		if state.dim() == 1:
			state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
		if action.dim() == 1:
			action = action.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
		elif action.dim() == 2:
			action = action.unsqueeze(1)  # (1, 1, action_dim)
		
		# 【关键修复】对于单步情况，不能用h[:, :-1, :]切片，会变成空张量
		# 解决方案：在单步推演时，直接使用encoder的输出
		with torch.no_grad():
			x = torch.cat([state, action], dim=-1)
			h, _ = self.encoder(x)  # (1, 1, hidden_dim)
			
			# 对于单步，h的形状是(1, 1, hidden_dim)，不需要切片
			z = self.to_latent(h[:, 0, :])  # (1, latent_dim) - 取序列第一个位置
			state_pred = self.state_decoder(z)  # (1, state_dim)
		
		return state_pred.squeeze()


# ===============================================
# 2. VLA模型 - 主治医师
# ===============================================
class MainPhysician(nn.Module):
	"""
	VLA模型 - 主治医师
	功能：
	1. 接收时序数据和文本提示
	2. 生成具体的治疗动作
	3. 生成自然语言诊断解释
	"""
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		hidden_dim: int = 128,
		latent_dim: int = 64,
		use_llm: bool = True,  # 是否使用LLM生成诊断
		llm_model: str = "glm-4-flash",  # LLM模型名称
		llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",  # 智谱API地址
		llm_api_key: Optional[str] = None,  # API Key（优先参数，其次环境变量）
		use_llm_verification: bool = False,  # 是否启用LLM二次验证
	):
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.use_llm = use_llm
		self.llm_model = llm_model
		self.llm_base_url = llm_base_url
		self.llm_api_key = llm_api_key or os.getenv("ZHIPU_API_KEY", "")
		self.use_llm_verification = use_llm_verification
		
		# 【新增】动作范围（在run()中设置）
		self.action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
		
		# 状态编码器
		self.state_encoder = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, latent_dim),
		)
		
		# 多头自注意力（处理维度间关系）
		self.attention = nn.MultiheadAttention(
			embed_dim=latent_dim,
			num_heads=4,
			dropout=0.1,
			batch_first=True
		)
		
		# 策略头：生成连续动作均值（Gaussian策略）
		self.policy_head = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Linear(hidden_dim // 2, action_dim),
			nn.Tanh(),  # 动作范围 [-1, 1]
		)
		
		# 策略头的标准差参数（可学习）
		self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
		
		# 值头：用于评估状态
		self.value_head = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Linear(hidden_dim // 2, 1),
		)
		
		# 高斯分布采样使用的初始化
		self.action_scale = 2.0  # 动作范围缩放

	def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		【优化】生成动作均值和价值估计
		现在支持序列输入，而不仅仅是单步
		
		Args:
			state: (batch, state_dim) 单步 或 (batch, seq_len, state_dim) 序列
			
		Returns:
			action_mean: (batch, action_dim) 高斯分布均值
			log_std: (action_dim,) 对数标准差
			value: (batch, 1)
		"""
		# 处理输入维度
		if state.dim() == 1:
			state = state.unsqueeze(0)  # (1, state_dim)
		
		if state.dim() == 2:
			# 单步：(batch, state_dim)
			latent = self.state_encoder(state)  # (batch, latent_dim)
			
			# 自注意力增强
			latent_att = latent.unsqueeze(1)  # (batch, 1, latent_dim)
			latent_att, _ = self.attention(latent_att, latent_att, latent_att)
			latent = latent + latent_att.squeeze(1)  # 残差连接
		
		elif state.dim() == 3:
			# 序列：(batch, seq_len, state_dim)
			# 【优化】对序列中的每个状态编码
			batch_size, seq_len, state_dim = state.shape
			state_flat = state.reshape(-1, state_dim)  # (batch*seq_len, state_dim)
			latent_flat = self.state_encoder(state_flat)  # (batch*seq_len, latent_dim)
			latent = latent_flat.reshape(batch_size, seq_len, -1)  # (batch, seq_len, latent_dim)
			
			# 时间池化：最后一帧或平均池化
			latent_att, _ = self.attention(latent, latent, latent)  # (batch, seq_len, latent_dim)
			latent = latent_att.mean(dim=1)  # (batch, latent_dim) - 时间平均
		
		# 策略和价值
		action_mean = self.policy_head(latent)  # 高斯分布均值
		value = self.value_head(latent)
		
		return action_mean, self.policy_log_std, value

	def generate_action(self, state: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		生成具体的连续动作
		
		Args:
			state: (state_dim,) 状态向量
			sample: 是否采样（否则取确定性动作）
			
		Returns:
			action: (action_dim,) 连续动作向量 【修改】已自动夹紧到训练范围内
			log_prob: 动作的对数概率密度 【修改】使用torch.distributions.Normal正确计算
		"""
		action_mean, log_std, _ = self.forward(state)
		
		# 【修改】使用torch.distributions.Normal获得正确的对数概率
		std = torch.exp(log_std)
		dist = Normal(action_mean, std)
		
		if sample:
			action = dist.rsample()  # 使用rsample以支持梯度
			log_prob = dist.log_prob(action).sum()
		else:
			# 确定性动作（均值）
			action = action_mean
			log_prob = torch.tensor(0.0, device=action.device)
		
		# 【新增】自动夹紧到训练数据的范围内，防止OOD
		if self.action_bounds is not None:
			action_min, action_max = self.action_bounds
			action = torch.clamp(action, min=action_min, max=action_max)
		
		# Tanh缩放到[-1,1]已在policy_head做了，这里不需要额外处理
		return action.squeeze(0), log_prob

	def generate_explanation(
		self, 
		state: torch.Tensor, 
		anomaly_score: float,
		action: Optional[torch.Tensor] = None
	) -> str:
		"""
		生成自然语言诊断
		
		Args:
			state: 当前状态
			anomaly_score: 异常程度 (0-1)
			action: 执行的连续动作（可选）
			
		Returns:
			explanation: 自然语言诊断文本
		"""
		# 动作分类名称（基于动作幅度）
		if action is not None:
			action_magnitude = torch.norm(action).item()
			if action_magnitude < 0.3:
				action_name = "轻微调整"
			elif action_magnitude < 0.6:
				action_name = "中等调整"
			else:
				action_name = "强烈干预"
		else:
			action_name = "待定动作"
		
		# 异常程度评估
		if anomaly_score > 0.8:
			severity = "严重"
		elif anomaly_score > 0.5:
			severity = "中等"
		else:
			severity = "轻微"
		
		# 状态分析（计算主要异常维度）
		state_np = state.cpu().detach().numpy() if isinstance(state, torch.Tensor) else state
		abnormal_dims = np.where(np.abs(state_np) > 2.0)[0]  # 标准差>2
		
		if len(abnormal_dims) > 0:
			reason = f"检测到{len(abnormal_dims)}个异常指标"
		else:
			reason = "多个指标联合异常"
		
		# 构建基础诊断信息
		base_info = (
			f"系统异常等级：{severity}（置信度 {anomaly_score:.1%}）\n"
			f"异常原因：{reason}\n"
			f"推荐动作：{action_name}"
		)
		
		# 如果启用LLM二次验证，先用LLM判断是否真的异常
		verification_info = ""
		if self.use_llm and self.use_llm_verification:
			try:
				verification_info = self._verify_anomaly_with_llm(state_np, reason, anomaly_score)
			except Exception as e:
				print(f"[警告] LLM验证失败: {e}")
		
		# 如果启用LLM，调用LLM生成详细诊断
		if self.use_llm:
			try:
				diagnosis = self._call_llm_for_diagnosis(base_info, action_name, anomaly_score, verification_info)
			except Exception as e:
				print(f"[警告] LLM调用失败: {e}，使用规则模板")
				diagnosis = self._generate_template_diagnosis(base_info, action_name, anomaly_score)
		else:
			diagnosis = self._generate_template_diagnosis(base_info, action_name, anomaly_score)
		
		return diagnosis

	def _verify_anomaly_with_llm(self, state: np.ndarray, reason: str, anomaly_score: float) -> str:
		"""
		LLM二次验证：判断检测到的异常是否真实
		
		Returns:
			verification_result: LLM的验证结论
		"""
		# 构建状态摘要
		top_abnormal_idx = np.argsort(np.abs(state))[-3:][::-1]  # 前3个异常维度
		top_abnormal_vals = state[top_abnormal_idx]
		
		state_summary = f"异常维度值: {', '.join([f'{v:.2f}' for v in top_abnormal_vals[:3]])}"
		
		verification_prompt = f"""你是工业系统故障诊断专家。请根据以下信息判断这是否为真实异常还是误报。

检测信息：
- 异常原因: {reason}
- 异常置信度: {anomaly_score:.1%}
- {state_summary}

请回答：
1. 这是真实异常吗？（是/否）
2. 简要理由（1句）

格式：是/否 | 理由"""
		
		if not self.llm_api_key:
			return ""

		try:
			api_url = self.llm_base_url.rstrip("/")
			if not api_url.endswith("/chat/completions"):
				api_url = f"{api_url}/chat/completions"

			response = requests.post(
				api_url,
				headers={
					"Authorization": f"Bearer {self.llm_api_key}",
					"Content-Type": "application/json",
				},
				json={
					"model": self.llm_model,
					"messages": [
						{"role": "system", "content": "你是工业故障诊断专家，判断要准确。"},
						{"role": "user", "content": verification_prompt},
					],
					"temperature": 0.2,  # 更低温度保证判断一致性
					"max_tokens": 50,
				},
				timeout=20
			)
			response.raise_for_status()
			result = response.json()
			choices = result.get("choices", [])
			if choices:
				verification = choices[0].get("message", {}).get("content", "")
				return f"【LLM二次验证】\n{verification}"
			return ""
		except Exception as e:
			print(f"[LLM验证错误] {e}")
			return ""

	def _call_llm_for_diagnosis(self, base_info: str, action_name: str, anomaly_score: float, verification_info: str = "") -> str:
		"""
		调用智谱清言API生成详细诊断
		"""
		# 如果有二次验证结果，包含在提示中
		verification_section = f"\n\n{verification_info}" if verification_info else ""
		
		prompt = f"""你是一个工业系统诊断专家。根据以下信息生成一份医学风格的诊断报告。

【系统信息】
{base_info}{verification_section}

请生成一份【诊断报告】，包括以下内容（保持简洁）：
1. 异常程度评估
2. 可能原因分析（1-2句）
3. 建议措施及原理
4. 预期效果

格式要求：
- 使用中文
- 总长度不超过200字
- 专业但易懂
"""
		
		if not self.llm_api_key:
			raise Exception("未配置ZHIPU_API_KEY，请先设置环境变量或传入--llm-api-key")

		try:
			api_url = self.llm_base_url.rstrip("/")
			if not api_url.endswith("/chat/completions"):
				api_url = f"{api_url}/chat/completions"

			response = requests.post(
				api_url,
				headers={
					"Authorization": f"Bearer {self.llm_api_key}",
					"Content-Type": "application/json",
				},
				json={
					"model": self.llm_model,
					"messages": [
						{"role": "system", "content": "你是工业系统故障诊断医生，回答要简洁、准确、可执行。"},
						{"role": "user", "content": prompt},
					],
					"temperature": 0.3,
					"max_tokens": 300,
				},
				timeout=30
			)
			response.raise_for_status()
			result = response.json()
			llm_output = ""
			choices = result.get("choices", [])
			if choices:
				llm_output = choices[0].get("message", {}).get("content", "")
			
			return f"【LLM诊断报告】\n{base_info}\n\n【AI分析】\n{llm_output}"
		except requests.exceptions.ConnectionError:
			raise Exception(f"无法连接到智谱API ({api_url})，请检查网络连接")
		except Exception as e:
			raise Exception(f"LLM生成失败: {str(e)}")

	def _generate_template_diagnosis(self, base_info: str, action_name: str, anomaly_score: float) -> str:
		"""
		生成模板诊断（LLM不可用时的备选方案）
		"""
		return (
			f"【诊断报告】\n"
			f"{base_info}\n\n"
			f"【系统提示】\n"
			f"已检测到系统异常。建议立即执行推荐动作。"
		)


@dataclass
class PreparedData:
	train_states: np.ndarray
	train_actions: np.ndarray
	test_states: np.ndarray
	test_actions: np.ndarray
	test_window_labels: np.ndarray
	state_cols: List[str]
	action_cols: List[str]


# ===============================================
# 3. RL框架 - 进化训练营
# ===============================================
class DreamEnvironment:
	"""
	手术模拟舱 - 梦境环境
	基于世界模型的虚拟环境，提供RL训练空间
	"""
	def __init__(
		self,
		world_model: HealthcheckSystem,
		state_mean: np.ndarray,
		state_std: np.ndarray,
		device: torch.device,
	):
		self.world_model = world_model
		self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=device)
		self.state_std = torch.tensor(state_std, dtype=torch.float32, device=device)
		self.device = device
		self.state_dim = world_model.state_dim
		self.action_dim = world_model.action_dim

	def step(
		self,
		state: torch.Tensor,
		action: torch.Tensor,
	) -> Tuple[torch.Tensor, float, bool]:
		"""
		在梦境中执行一步
		
		Args:
			state: (state_dim,) 当前状态
			action: (action_dim,) 或 (num_actions,) 执行的动作
			
		Returns:
			next_state: 下一状态
			reward: 奖励值
			done: 是否终止
		"""
		# 确保在正确的设备上
		state = state.to(self.device)
		action = action.to(self.device)
		
		# 模拟下一步
		next_state = self.world_model.simulate_step(state, action, device=self.device)
		
		# 计算奖励
		reward = self.compute_reward(state, next_state, action)
		
		# 检查是否崩溃
		done = self.check_collapse(next_state)
		
		return next_state, reward, done

	def compute_reward(
		self,
		state: torch.Tensor,
		next_state: torch.Tensor,
		action: torch.Tensor
	) -> float:
		"""
		【修复】改用Z-Score幅度判定异常（避免自参考问题）
		
		由于数据已标准化（Z-Score Normalization），正常状态值在0附近。
		异常时数值会飙升到 > 2.0（2σ）或 > 3.0（3σ）。
		
		奖励定义：
		- 异常明显缓解（>2σ的维度减少>20%）：+1.0
		- 异常略缓解（>2σ的维度减少5-20%）：+0.1
		- 无改善或略恶化：0.0 ~ -0.1
		- 异常严重恶化：-0.5
		- 系统崩溃（>50%维度超3σ）：-10.0
		"""
		next_state_np = next_state.cpu().numpy()
		state_np = state.cpu().numpy()
		
		# 检查系统是否崩溃（>50%维度超3σ）
		collapse_ratio = np.mean(np.abs(next_state_np) > 3.0)
		if collapse_ratio > 0.5:
			return -10.0
		
		# 【关键】使用Z-Score幅度作为异常指标
		# 计算当前和下一步中超过2σ的维度比例
		current_abnormal = np.mean(np.abs(state_np) > 2.0)  # 正常0附近，异常>2.0
		next_abnormal = np.mean(np.abs(next_state_np) > 2.0)
		
		# 异常改善程度（比例变化）
		abnormal_reduction = current_abnormal - next_abnormal
		
		# 检测OOD动作惩罚
		action_ood_penalty = self._compute_ood_penalty(action)
		
		# 分级奖励
		if abnormal_reduction > 0.2:  # 异常维度减少>20%
			return 1.0 - action_ood_penalty
		elif abnormal_reduction > 0.05:  # 异常维度减少5-20%
			return 0.1 - action_ood_penalty
		elif abnormal_reduction < -0.1:  # 异常维度增加>10%
			return -0.5 - action_ood_penalty
		else:  # 基本无改变
			return -action_ood_penalty

	def _compute_ood_penalty(self, action: torch.Tensor) -> float:
		"""计算超出训练分布的动作惩罚（0到0.5）"""
		if self.world_model.action_min is None or self.world_model.action_max is None:
			return 0.0
		
		action = action.squeeze()
		out_of_bounds = (action < self.world_model.action_min) | (action > self.world_model.action_max)
		violation_count = out_of_bounds.sum().item()
		violation_ratio = violation_count / len(action)
		
		if violation_ratio > 0:
			# 超出范围的维度越多，惩罚越大（0-0.5）
			return min(0.5, violation_ratio * 0.5)
		return 0.0

	def check_collapse(self, state: torch.Tensor) -> bool:
		"""检查系统是否崩溃"""
		state_np = state.cpu().detach().numpy()
		anomaly_ratio = np.mean(np.abs(state_np) > 3.0)  # 超过3倍标准差
		return bool(anomaly_ratio > 0.5)


class RLTrainer:
	"""
	RL训练器 - 进化训练营
	使用GRPO或PPO进行策略优化
	"""
	def __init__(
		self,
		policy: MainPhysician,
		world_model: HealthcheckSystem,
		device: torch.device,
		lr: float = 1e-3,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		clip_ratio: float = 0.2,
		entropy_coef: float = 0.01,
	):
		self.policy = policy
		self.world_model = world_model
		self.device = device
		self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_ratio = clip_ratio
		self.entropy_coef = entropy_coef

	def collect_trajectory(
		self,
		initial_state: torch.Tensor,
		environment: DreamEnvironment,
		horizon: int = 20,
	) -> Dict[str, np.ndarray]:
		"""
		在梦境中收集一条轨迹
		
		Args:
			initial_state: 初始状态（应已在device上）
			environment: 梦境环境
			horizon: 轨迹长度
			
		Returns:
			trajectory: 包含states, actions, rewards, values, dones
		"""
		states = []
		actions = []
		rewards = []
		values = []
		dones = []
		log_probs = []
		
		# 【修复Device问题】确保初始状态在正确device上
		state = initial_state.clone().to(self.device)
		self.policy.eval()
		
		with torch.no_grad():
			for t in range(horizon):
				states.append(state.cpu().numpy())
				
				# 【修复】保持单步一致性 - 删除state_buffer复杂逻辑
				# 采集时用单步，更新时也用单步 → 观测空间一致
				action_mean, log_std, value = self.policy(state.unsqueeze(0))  # (1, state_dim)
				
				values.append(value.item())
				
				# 【使用】torch.distributions.Normal采样动作并计算对数概率
				std = torch.exp(log_std)
				dist = Normal(action_mean, std)
				action = dist.rsample()
				log_prob = dist.log_prob(action).sum()
				log_probs.append(log_prob.item())
				
				action = action.squeeze(0).to(self.device)
				actions.append(action.cpu().numpy())
				
				# 执行步骤
				next_state, reward, done = environment.step(state, action)
				
				rewards.append(reward)
				dones.append(done)
				
				state = next_state
				if done:
					break
		
		# 计算优势（GAE）
		advantages, returns = self.compute_gae(
			np.array(rewards),
			np.array(values),
			np.array(dones)
		)
		
		return {
			'states': np.array(states),
			'actions': np.array(actions),
			'rewards': np.array(rewards),
			'values': np.array(values),
			'advantages': advantages,
			'returns': returns,
			'log_probs': np.array(log_probs),
			'dones': np.array(dones),
		}

	def compute_gae(
		self,
		rewards: np.ndarray,
		values: np.ndarray,
		dones: np.ndarray
	) -> Tuple[np.ndarray, np.ndarray]:
		"""计算广义优势估计（GAE）"""
		advantages = []
		advantage = 0
		
		for t in reversed(range(len(rewards))):
			if t == len(rewards) - 1:
				next_value = 0
			else:
				next_value = values[t + 1] * (1 - dones[t])
			
			td_error = rewards[t] + self.gamma * next_value - values[t]
			advantage = td_error + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
			advantages.insert(0, advantage)
		
		advantages = np.array(advantages)
		returns = advantages + values
		
		# 归一化优势
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		
		return advantages, returns

	def update_policy(self, trajectories: List[Dict]) -> float:
		"""
		使用轨迹数据更新策略
		"""
		self.policy.train()
		
		total_loss = 0.0
		num_updates = 0
		
		for trajectory in trajectories:
			states = torch.tensor(trajectory['states'], dtype=torch.float32, device=self.device)
			actions = torch.tensor(trajectory['actions'], dtype=torch.float32, device=self.device)
			old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float32, device=self.device)
			advantages = torch.tensor(trajectory['advantages'], dtype=torch.float32, device=self.device)
			returns = torch.tensor(trajectory['returns'], dtype=torch.float32, device=self.device)
			
			# PPO更新
			for _ in range(3):  # 3次内部epoch
				action_mean, log_std, values = self.policy(states)
				
				# 【修改】使用torch.distributions.Normal正确计算高斯分布的对数概率
				std = torch.exp(log_std.unsqueeze(0).expand_as(action_mean))
				dist = Normal(action_mean, std)
				log_probs_new = dist.log_prob(actions).sum(dim=1)
				
				ratio = torch.exp(log_probs_new - old_log_probs)
				surr1 = ratio * advantages
				surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
				policy_loss = -torch.min(surr1, surr2).mean()
				
				# 价值损失
				value_loss = F.smooth_l1_loss(values.squeeze(1), returns)
				
				# 总损失
				loss = policy_loss + 0.5 * value_loss
				
				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
				self.optimizer.step()
				
				total_loss += loss.item()
				num_updates += 1
		
		return total_loss / max(num_updates, 1)


def prepare_hai_data(
	version_dir: str,
	window_size: int,
	stride: int,
	max_train_rows_per_file: Optional[int] = None,
	max_test_rows_per_file: Optional[int] = None,
) -> Tuple[PreparedData, np.ndarray, np.ndarray]:
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

	prepared = PreparedData(
		train_states=train_state_windows,
		train_actions=train_action_windows,
		test_states=test_state_windows,
		test_actions=test_action_windows,
		test_window_labels=test_window_labels,
		state_cols=state_cols,
		action_cols=action_cols,
	)
	
	return prepared, state_mean.squeeze(), state_std.squeeze()


# ===============================================
# 训练函数
# ===============================================
def train_healthcheck_system(
	model: HealthcheckSystem,
	train_states: np.ndarray,
	train_actions: np.ndarray,
	batch_size: int,
	epochs: int,
	lr: float,
	device: torch.device,
) -> None:
	"""
	训练世界模型（体检仪）
	
	【优化】新增Early Stopping防止过拟合
	- 将训练集80/20分为train/val
	- 监控验证集损失
	- 如果val_loss连续patience轮不下降，则停止训练
	"""
	# 【新增】分割训练集和验证集（80/20）
	n_samples = len(train_states)
	split_idx = int(0.8 * n_samples)
	
	train_states_split = train_states[:split_idx]
	train_actions_split = train_actions[:split_idx]
	val_states = train_states[split_idx:]
	val_actions = train_actions[split_idx:]
	
	ds_train = SequenceDataset(train_states_split, train_actions_split)
	ds_val = SequenceDataset(val_states, val_actions)
	
	loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
	loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

	# 【改进】增加weight_decay实现L2正则化
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
	criterion = nn.MSELoss()
	model.to(device)

	print(f"\n【训练体检仪】device={device}, train_windows={len(ds_train)}, val_windows={len(ds_val)}, epochs={epochs}")
	
	# 【新增】Early Stopping参数
	best_val_loss = float('inf')
	patience = 5
	patience_counter = 0
	best_model_state = None
	
	for epoch in range(1, epochs + 1):
		model.train()
		running_state = 0.0
		running_action = 0.0
		steps = 0
		
		for states, actions in loader_train:
			states = states.to(device)
			actions = actions.to(device)
			
			state_pred, action_pred, state_error = model(states, actions)
			state_target = states[:, 1:, :]
			action_target = actions[:, 1:, :]
			
			state_loss = criterion(state_pred, state_target)
			action_loss = criterion(action_pred, action_target)
			
			loss = state_loss + 0.5 * action_loss

			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			running_state += state_loss.item()
			running_action += action_loss.item()
			steps += 1

		# 【新增】验证集评估
		model.eval()
		val_state_loss = 0.0
		val_action_loss = 0.0
		val_steps = 0
		
		with torch.no_grad():
			for states, actions in loader_val:
				states = states.to(device)
				actions = actions.to(device)
				
				state_pred, action_pred, _ = model(states, actions)
				state_target = states[:, 1:, :]
				action_target = actions[:, 1:, :]
				
				state_loss = criterion(state_pred, state_target)
				action_loss = criterion(action_pred, action_target)
				
				val_state_loss += state_loss.item()
				val_action_loss += action_loss.item()
				val_steps += 1
		
		avg_val_loss = (val_state_loss + 0.5 * val_action_loss) / max(1, val_steps)
		
		# 【新增】Early Stopping逻辑
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
			best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
		else:
			patience_counter += 1
		
		if epoch % max(1, epochs // 10) == 0 or epoch == 1:
			print(f"  epoch={epoch:03d}/{epochs} train_state_loss={running_state/max(1,steps):.6f} train_action_loss={running_action/max(1,steps):.6f} val_loss={avg_val_loss:.6f} patience={patience_counter}/{patience}")
		
		# 【新增】当patience耗尽时，停止训练并恢复最佳模型
		if patience_counter >= patience:
			print(f"  【Early Stopping】在epoch={epoch}停止，验证损失连续{patience}轮未改进")
			if best_model_state is not None:
				model.load_state_dict(best_model_state)
				print(f"  已恢复最佳模型（val_loss={best_val_loss:.6f}）")
			break


def collect_health_scores(
	model: HealthcheckSystem,
	states: np.ndarray,
	actions: np.ndarray,
	batch_size: int,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	收集体检仪的检查分数
	
	【修改】删除冗余的anomaly_head，直接用重构误差作为异常指标
	"""
	ds = SequenceDataset(states, actions)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

	model.eval()
	state_out = []
	action_out = []
	
	with torch.no_grad():
		for batch_states, batch_actions in loader:
			batch_states = batch_states.to(device)
			batch_actions = batch_actions.to(device)
			
			# forward现在返回 (state_pred, action_pred, state_error)
			state_pred, action_pred, state_error = model(batch_states, batch_actions)
			state_target = batch_states[:, 1:, :]
			action_target = batch_actions[:, 1:, :]
			
			state_err = torch.mean((state_pred - state_target) ** 2, dim=(1, 2))
			action_err = torch.mean((action_pred - action_target) ** 2, dim=(1, 2))
			
			state_out.append(state_err.cpu().numpy())
			action_out.append(action_err.cpu().numpy())
	
	state_scores = np.concatenate(state_out, axis=0)
	action_scores = np.concatenate(action_out, axis=0)
	
	# 【修改】综合异常分数：70%来自state重构误差，30%来自action预测误差
	# （删掉了冗余的anomaly_head）
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

	print("=" * 60)
	print("【医疗架构异常检测系统】")
	print("体检仪（异常检测）→ 主治医师（动作生成）→ 进化训练营（策略优化）")
	print("=" * 60)

	# 1. 数据加载
	print("\n【第一步】加载数据...")
	data, state_mean, state_std = prepare_hai_data(
		version_dir=args.data_dir,
		window_size=args.window,
		stride=args.stride,
		max_train_rows_per_file=args.max_train_rows,
		max_test_rows_per_file=args.max_test_rows,
	)

	print(f"  state_dim={len(data.state_cols)}, action_dim={len(data.action_cols)}")
	print(f"  train: states={data.train_states.shape}, actions={data.train_actions.shape}")
	print(f"  test: states={data.test_states.shape}, actions={data.test_actions.shape}")
	print(f"  test_anomaly_ratio={data.test_window_labels.mean():.4f}")

	# 【新增】计算动作范围，用于distribution shift检测
	action_min = data.train_actions.min(axis=0)
	action_max = data.train_actions.max(axis=0)
	print(f"  动作范围（训练集）: min={action_min[:3]}..., max={action_max[:3]}...")

	# 2. 构建模型
	print("\n【第二步】构建医疗系统...")
	
	# 体检仪（世界模型）
	healthcheck = HealthcheckSystem(
		state_dim=len(data.state_cols),
		action_dim=len(data.action_cols),
		hidden_dim=args.hidden,
		latent_dim=args.latent,
		num_layers=args.layers,
		dropout=args.dropout,
		action_min=action_min,  # 【新增】传递动作范围
		action_max=action_max,  # 【新增】传递动作范围
	)
	
	# 主治医师（VLA模型）
	physician = MainPhysician(
		state_dim=len(data.state_cols),
		action_dim=len(data.action_cols),
		hidden_dim=args.hidden,
		latent_dim=args.latent,
		use_llm=args.use_llm_diagnosis,
		llm_model=args.llm_model,
		llm_base_url=args.llm_base_url,
		llm_api_key=args.llm_api_key,
		use_llm_verification=args.use_llm_verification,
	)
	
	# 【新增】设置动作范围，用于自动夹紧（防止OOD）
	physician.action_bounds = (
		torch.tensor(action_min, dtype=torch.float32),
		torch.tensor(action_max, dtype=torch.float32)
	)
	
	print("  ✓ 体检仪已构建（世界模型）")
	print("  ✓ 主治医师已构建（VLA模型）")

	# 3. 训练体检仪
	print("\n【第三步】训练体检仪...")
	train_healthcheck_system(
		model=healthcheck,
		train_states=data.train_states,
		train_actions=data.train_actions,
		batch_size=args.batch,
		epochs=args.epochs,
		lr=args.lr,
		device=device,
	)

	# 4. 异常检测评估
	print("\n【第四步】体检仪异常检测评估...")
	train_state_scores, train_action_scores, train_combined = collect_health_scores(
		healthcheck, data.train_states, data.train_actions, args.batch, device
	)
	test_state_scores, test_action_scores, test_combined = collect_health_scores(
		healthcheck, data.test_states, data.test_actions, args.batch, device
	)

	threshold = float(np.percentile(train_combined, args.threshold_percentile))
	y_pred = (test_combined > threshold).astype(np.int64)

	metrics = binary_metrics(data.test_window_labels, y_pred)

	print("\n【体检仪检查结果】")
	print(f"  阈值设置: {threshold:.6f} (第{args.threshold_percentile}百分位)")
	print(f"  精准度(Precision): {metrics['precision']:.4f}")
	print(f"  召回率(Recall): {metrics['recall']:.4f}")
	print(f"  F1分数: {metrics['f1']:.4f}")
	print(f"  混淆矩阵: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
	
	# 5. 如果启用RL训练
	if args.use_rl:
		print("\n【第五步】进化训练营 - RL策略优化...")
		print("  构建梦境环境（手术模拟舱）...")
		
		dream_env = DreamEnvironment(
			world_model=healthcheck,
			state_mean=state_mean,
			state_std=state_std,
			device=device,
		)
		
		print("  初始化主治医师进化训练...")
		trainer = RLTrainer(
			policy=physician,
			world_model=healthcheck,
			device=device,
			lr=args.rl_lr,
			gamma=args.gamma,
			gae_lambda=args.gae_lambda,
			clip_ratio=args.clip_ratio,
			entropy_coef=args.entropy_coef,
		)
		
		# 收集梦境轨迹
		print(f"  开始梦境模拟 ({args.rl_epochs} 轮)...")
		for rl_epoch in range(args.rl_epochs):
			trajectories = []
			
			# 从测试集中随机选择初始状态进行梦境模拟
			for _ in range(args.rl_episodes):
				idx = np.random.randint(0, len(data.test_states))
				initial_state = torch.tensor(data.test_states[idx, 0], dtype=torch.float32, device=device)
				
				traj = trainer.collect_trajectory(
					initial_state=initial_state,
					environment=dream_env,
					horizon=args.rl_horizon,
				)
				trajectories.append(traj)
			
			# 更新策略
			loss = trainer.update_policy(trajectories)
			
			if (rl_epoch + 1) % max(1, args.rl_epochs // 5) == 0 or rl_epoch == 0:
				print(f"    梦境轮 {rl_epoch+1}/{args.rl_epochs}, 策略损失={loss:.6f}")
		
		print("  ✓ 主治医师已完成进化训练！")

	# 6. 演示医师诊断
	print("\n【第六步】主治医师诊断演示...")
	demo_idx = np.where(test_combined > threshold)[0]
	if len(demo_idx) > 0:
		idx = demo_idx[0]
		state_sample = torch.tensor(data.test_states[idx, 0], dtype=torch.float32)
		anomaly_score_sample = float(test_combined[idx])
		
		# 生成示例动作进行诊断
		with torch.no_grad():
			action, _ = physician.generate_action(state_sample, sample=True)
		
		diagnosis = physician.generate_explanation(state_sample, anomaly_score_sample, action)
		print(diagnosis)

	# 7. 保存模型
	if args.save_model:
		os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
		ckpt = {
			"healthcheck_state": healthcheck.state_dict(),
			"physician_state": physician.state_dict(),
			"state_cols": data.state_cols,
			"action_cols": data.action_cols,
			"window": args.window,
			"stride": args.stride,
			"threshold": threshold,
			"state_mean": state_mean,
			"state_std": state_std,
		}
		torch.save(ckpt, args.save_model)
		print(f"\n【保存完成】模型已保存: {args.save_model}")

	print("\n" + "=" * 60)
	print("【系统运行完毕】")
	print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="医疗架构异常检测系统 - 体检仪+手术模拟舱+主治医师+进化训练营"
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
	parser.add_argument("--epochs", type=int, default=5, help="体检仪训练轮数")
	parser.add_argument("--lr", type=float, default=1e-3, help="体检仪学习率")
	parser.add_argument("--threshold-percentile", type=float, default=99.8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--use-rl", action="store_true", help="启用RL进化训练营")
	parser.add_argument("--rl-lr", type=float, default=1e-3, help="RL学习率")
	parser.add_argument("--rl-epochs", type=int, default=10, help="RL训练轮数")
	parser.add_argument("--rl-episodes", type=int, default=5, help="每轮RL梦境轨迹数")
	parser.add_argument("--rl-horizon", type=int, default=20, help="梦境轨迹长度")
	parser.add_argument("--gamma", type=float, default=0.99, help="RL折扣因子")
	parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE参数")
	parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO裁剪比")
	parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵系数")
	
	# LLM诊断相关参数
	parser.add_argument("--use-llm-diagnosis", action="store_true", help="启用LLM生成详细诊断")
	parser.add_argument("--llm-model", type=str, default="glm-4-flash", help="LLM模型名称 (默认: glm-4-flash)")
	parser.add_argument(
		"--llm-base-url",
		type=str,
		default="https://open.bigmodel.cn/api/paas/v4",
		help="LLM API地址 (默认: 智谱清言Chat Completions)",
	)
	parser.add_argument(
		"--llm-api-key",
		type=str,
		default=None,
		help="LLM API密钥（不传则读取环境变量ZHIPU_API_KEY）",
	)
	parser.add_argument(
		"--use-llm-verification",
		action="store_true",
		help="启用LLM二次验证：判断检测到的异常是否真实"
	)
	
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