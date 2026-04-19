import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


SEED = 42


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(SEED)


# ==========================================
# 1. 数据生成 (Toy Dataset)
# ==========================================
def generate_synthetic_data(length=5000, anomaly_ratio=0.05):
    rng = np.random.default_rng(SEED if length == 5000 else SEED + 1000)
    t = np.linspace(0, 50, length)
    data = np.sin(t) + rng.normal(0, 0.1, length)
    labels = np.zeros(length)
    num_anomalies = int(length * anomaly_ratio)
    anomaly_indices = rng.choice(length, num_anomalies, replace=False)
    for idx in anomaly_indices:
        data[idx] += rng.choice([3.0, -3.0])
        labels[idx] = 1.0
    return data, labels

train_data, train_labels = generate_synthetic_data(length=5000)
test_data, test_labels = generate_synthetic_data(length=2000)
window_size = 10

scaler = StandardScaler()
# 在生成数据后立即进行归一化
train_data = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_data = scaler.transform(test_data.reshape(-1, 1)).flatten()

def extract_windows(data, window_size):
    windows =[]
    for i in range(window_size - 1, len(data)):
        windows.append(data[i - window_size + 1 : i + 1])
    return np.array(windows)

# ==========================================
# 2. 定义与预训练 VAE 模型
# ==========================================
class VAE(nn.Module):
    def __init__(self, window_size, latent_dim=3):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(window_size, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, window_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def evaluate_vae_baseline(data, labels, vae, threshold, window_size):
    windows = extract_windows(data, window_size)
    truth = labels[window_size - 1:]
    model_input = torch.FloatTensor(windows)
    vae.eval()
    with torch.no_grad():
        recon_windows, _, _ = vae(model_input)
        mse_scores = torch.mean((recon_windows - model_input) ** 2, dim=1).numpy()
    preds = (mse_scores > threshold).astype(int)
    return {
        "precision": precision_score(truth, preds, zero_division=0),
        "recall": recall_score(truth, preds, zero_division=0),
        "f1": f1_score(truth, preds),
    }

print("--- 步骤 1: 开始预训练 VAE ---")
train_windows = extract_windows(train_data, window_size)
train_tensor = torch.FloatTensor(train_windows)

vae = VAE(window_size)
vae_optimizer = optim.Adam(vae.parameters(), lr=0.005)
vae.train()

for epoch in range(20):
    vae_optimizer.zero_grad()
    recon_batch, mu, logvar = vae(train_tensor)
    loss = vae_loss_function(recon_batch, train_tensor, mu, logvar)
    loss.backward()
    vae_optimizer.step()

# 计算阈值
vae.eval()
with torch.no_grad():
    recon_train, _, _ = vae(train_tensor)
    mse_errors = torch.mean((recon_train - train_tensor)**2, dim=1).numpy()

mean_error = np.mean(mse_errors)
std_error = np.std(mse_errors)
threshold = np.percentile(mse_errors, 95)
pseudo_anomaly_rate = float(np.mean(mse_errors > threshold))
anomaly_reward_weight = float(min(8.0, max(2.0, (1.0 - pseudo_anomaly_rate) / max(pseudo_anomaly_rate, 1e-6))))
print(f"VAE 预训练完成! MSE 均值: {mean_error:.4f}, 标准差: {std_error:.4f}, 95分位阈值: {threshold:.4f}\n")
print(f"伪异常占比: {pseudo_anomaly_rate:.3f}, 异常奖励权重: {anomaly_reward_weight:.2f}\n")

vae_baseline_test = evaluate_vae_baseline(test_data, test_labels, vae, threshold, window_size)
print(
    "VAE 直接阈值基线 - "
    f"Test Precision: {vae_baseline_test['precision']:.3f}, "
    f"Recall: {vae_baseline_test['recall']:.3f}, "
    f"F1-Score: {vae_baseline_test['f1']:.3f}\n"
)

# ==========================================
# 3. 基于 VAE 的无监督强化学习环境
# ==========================================
class VaeRewardEnv:
    def __init__(self, data, labels, vae, threshold, window_size=10, anomaly_reward_weight=3.0):
        self.data = data
        self.labels = labels
        self.vae = vae
        self.threshold = threshold
        self.window_size = window_size
        self.anomaly_reward_weight = anomaly_reward_weight
        self.current_step = window_size - 1
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = self.window_size - 1
        return self._get_obs()

    def _get_obs(self):
        obs = self.data[self.current_step - self.window_size + 1 : self.current_step + 1]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        state = self._get_obs()
        true_label = self.labels[self.current_step] # 仅用于测试评估，不用于生成奖励！
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            recon_state, _, _ = self.vae(state_tensor)
            mse = torch.mean((recon_state - state_tensor)**2).item()
        
        # 为了避免策略塌缩到“全部判正常”，异常样本使用更高权重
        is_anomalous = mse > self.threshold  # True 表示异常

        if is_anomalous:
            reward = self.anomaly_reward_weight if action == 1 else -self.anomaly_reward_weight
        else:
            reward = 1.0 if action == 0 else -1.0
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_obs() if not done else np.zeros(self.window_size, dtype=np.float32)
        
        return next_state, reward, done, true_label

# ==========================================
# 4. 定义 DQN 与训练循环
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

state_dim = window_size
action_dim = 2
learning_rate = 0.005
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.95
epsilon_min = 0.05
batch_size = 32

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()
replay_buffer = deque(maxlen=2000)
target_update_freq = 200
train_step_count = 0

env = VaeRewardEnv(
    train_data,
    train_labels,
    vae,
    threshold,
    window_size,
    anomaly_reward_weight=anomaly_reward_weight,
)

print("--- 步骤 2: 开始基于 VAE 奖励的 DQN 无监督训练 ---")
epochs = 20
for epoch in range(epochs):
    state = env.reset()
    done = False
    total_reward = 0
    q_net.train()
    
    while not done:
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item()
        
        next_state, reward, done, _ = env.step(action) # 注意：这里没有用到真实标签发奖励
        total_reward += reward
        
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
            
            b_states = torch.FloatTensor(np.array(b_states))
            b_actions = torch.LongTensor(b_actions).unsqueeze(1)
            b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1)
            b_next_states = torch.FloatTensor(np.array(b_next_states))
            b_dones = torch.FloatTensor(b_dones).unsqueeze(1)
            
            current_q = q_net(b_states).gather(1, b_actions)
            with torch.no_grad():
                max_next_q = target_net(b_next_states).max(1)[0].unsqueeze(1)
                target_q = b_rewards + (1 - b_dones) * gamma * max_next_q
                
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
            optimizer.step()
            train_step_count += 1
            if train_step_count % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
            
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Epoch {epoch+1}/{epochs} | Total Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

# ==========================================
# 5. 测试与评估
# ==========================================
print("\n--- 步骤 3: 测试无监督 RL 效果 ---")
test_env = VaeRewardEnv(
    test_data,
    test_labels,
    vae,
    threshold,
    window_size,
    anomaly_reward_weight=anomaly_reward_weight,
)
state = test_env.reset()
done = False

preds_rl =[]
truths =[]

q_net.eval()

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
        action = torch.argmax(q_values).item()
    
    next_state, _, done, true_label = test_env.step(action)
    
    preds_rl.append(action)
    truths.append(true_label)
    state = next_state

# 计算 RL 模型的指标
rl_f1 = f1_score(truths, preds_rl)
rl_prec = precision_score(truths, preds_rl, zero_division=0)
rl_rec = recall_score(truths, preds_rl, zero_division=0)
rl_positive_rate = float(np.mean(preds_rl))

print(f"RL 模型 (DQN) - Precision: {rl_prec:.3f}, Recall: {rl_rec:.3f}, F1-Score: {rl_f1:.3f}\n")
print(f"RL 预测为异常的比例: {rl_positive_rate:.3f}\n")


# ==========================================
# 6. 传统无监督 Baseline (Isolation Forest)
# ==========================================
from sklearn.ensemble import IsolationForest

print("--- 开始传统无监督 Baseline (Isolation Forest) 对比 ---")
train_windows = extract_windows(train_data, window_size)
test_windows = extract_windows(test_data, window_size)

# 训练孤立森林 (无监督，不需要 labels)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(train_windows)

# 预测测试集 (-1 为异常，1 为正常，需要转换为 1 和 0)
iso_preds_raw = iso_forest.predict(test_windows)
preds_iso = [1 if p == -1 else 0 for p in iso_preds_raw]

# 注意：提取窗口后，标签从 window_size-1 开始
iso_truths = test_labels[window_size-1:]

iso_f1 = f1_score(iso_truths, preds_iso)
iso_prec = precision_score(iso_truths, preds_iso, zero_division=0)
iso_rec = recall_score(iso_truths, preds_iso, zero_division=0)

print(f"Isolation Forest - Precision: {iso_prec:.3f}, Recall: {iso_rec:.3f}, F1-Score: {iso_f1:.3f}")

print("\n--- 第二阶段目标全部达成！ ---")