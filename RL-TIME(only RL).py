import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd
except ImportError:
    pd = None

# ==========================================
# 1. 数据生成 (与之前相同)
# ==========================================
def generate_synthetic_data(length=5000, anomaly_ratio=0.05):
    np.random.seed(42)
    t = np.linspace(0, 50, length)
    data = np.sin(t) + np.random.normal(0, 0.1, length)
    labels = np.zeros(length)
    num_anomalies = int(length * anomaly_ratio)
    anomaly_indices = np.random.choice(length, num_anomalies, replace=False)
    for idx in anomaly_indices:
        data[idx] += np.random.choice([3.0, -3.0])
        labels[idx] = 1.0
    return data, labels


def load_real_data(csv_path, label_col="label", train_ratio=0.7):
    if pd is None:
        raise ImportError("需要 pandas 才能加载真实CSV数据，请先安装 pandas。")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"在 {csv_path} 中找不到标签列: {label_col}")

    labels = df[label_col].astype(int).to_numpy()
    feature_cols = [c for c in df.columns if c != label_col]
    if not feature_cols:
        raise ValueError("标签列之外没有可用特征列。")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    split_idx = int(len(features) * train_ratio)
    if split_idx <= 0 or split_idx >= len(features):
        raise ValueError("train_ratio 设置不合法，请使用 (0, 1) 之间的值。")

    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[:split_idx])
    test_features = scaler.transform(features[split_idx:])

    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    return train_features, train_labels, test_features, test_labels


USE_REAL_DATA = False
REAL_DATA_PATH = "walmart_cleaned.csv"
LABEL_COL = "label"

if USE_REAL_DATA:
    train_data, train_labels, test_data, test_labels = load_real_data(
        REAL_DATA_PATH,
        label_col=LABEL_COL,
        train_ratio=0.7,
    )
else:
    train_data, train_labels = generate_synthetic_data(length=5000)
    test_data, test_labels = generate_synthetic_data(length=2000)

# ==========================================
# 2. 纯手写环境 (极简版)
# ==========================================
class SimpleEnv:
    def __init__(self, data, labels, window_size=10):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = self.window_size
        return self._get_obs()

    def _get_obs(self):
        obs = self.data[self.current_step - self.window_size : self.current_step]
        if obs.ndim == 2:
            obs = obs.reshape(-1)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        true_label = self.labels[self.current_step]
        
        # 奖励函数设计 (你未来就在这里魔改，加入大模型和VAE的信号)
        if action == true_label:
            reward = 1.0
        else:
            reward = -5.0 if true_label == 1 else -1.0
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_obs() if not done else np.zeros(self.window_size, dtype=np.float32)
        
        return next_state, reward, done, true_label

# ==========================================
# 3. 定义 PyTorch 神经网络 (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 一个简单的两层全连接网络
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_dim) # 输出两个值：Q(s, 0) 和 Q(s, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# 4. DQN 智能体与训练循环
# ==========================================
# 超参数设定
window_size = 10
feature_dim = train_data.shape[1] if np.ndim(train_data) == 2 else 1
state_dim = window_size * feature_dim
action_dim = 2
learning_rate = 0.001
gamma = 0.99       # 奖励折扣因子
epsilon = 1.0      # 初始探索率 (100% 随机猜)
epsilon_decay = 0.995 # 探索率衰减
epsilon_min = 0.01
batch_size = 64

# 初始化网络和优化器
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 经验回放池 (Replay Buffer)
replay_buffer = deque(maxlen=2000)

env = SimpleEnv(train_data, train_labels, window_size)

print("--- 开始纯 PyTorch DQN 训练 ---")
epochs = 3 # 遍历数据集几遍
for epoch in range(epochs):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 1. 探索与利用 (Epsilon-Greedy)
        if random.random() < epsilon:
            action = random.choice([0, 1]) # 随机探索
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item() # 选择Q值最大的动作
        
        # 2. 与环境交互
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # 3. 将经验存入回放池
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        # 4. 从回放池中抽样并学习 (核心 RL 算法)
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
            
            b_states = torch.FloatTensor(np.array(b_states))
            b_actions = torch.LongTensor(b_actions).unsqueeze(1)
            b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1)
            b_next_states = torch.FloatTensor(np.array(b_next_states))
            b_dones = torch.FloatTensor(b_dones).unsqueeze(1)
            
            # 计算当前 Q 值: Q(s, a)
            current_q = q_net(b_states).gather(1, b_actions)
            
            # 计算目标 Q 值: r + gamma * max(Q(s', a'))
            with torch.no_grad():
                max_next_q = q_net(b_next_states).max(1)[0].unsqueeze(1)
                target_q = b_rewards + (1 - b_dones) * gamma * max_next_q
                
            # 计算损失并反向传播
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 每跑完一遍数据集，降低探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

print("训练完成！\n")

# ==========================================
# 5. 测试评估
# ==========================================
print("--- 开始测试评估 ---")
test_env = SimpleEnv(test_data, test_labels, window_size)
state = test_env.reset()
done = False

preds = []
truths =[]

q_net.eval()

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
        action = torch.argmax(q_values).item()
    
    next_state, _, done, true_label = test_env.step(action)
    preds.append(action)
    truths.append(true_label)
    state = next_state

f1 = f1_score(truths, preds)
precision = precision_score(truths, preds, zero_division=0)
recall = recall_score(truths, preds, zero_division=0)
pred_anomaly_ratio = float(np.mean(preds))
true_anomaly_ratio = float(np.mean(truths))

print(f"Pure RL - Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
print(f"预测异常比例: {pred_anomaly_ratio:.3f} | 真实异常比例: {true_anomaly_ratio:.3f}")