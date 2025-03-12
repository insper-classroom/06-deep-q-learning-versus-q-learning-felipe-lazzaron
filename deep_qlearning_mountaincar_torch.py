#!/usr/bin/env python3
"""
Deep Q-Learning para o ambiente MountainCar-v0 usando PyTorch

Atividades:
- Implementa Deep Q-Learning para o ambiente MountainCar-v0.
- Executa o treinamento N vezes (N >= 5) para coletar os dados da curva de aprendizado.
- Salva os pesos da rede neural para cada execução.
- Utiliza os hiperparâmetros do exemplo fornecido.

Hiperparâmetros (baseados no exemplo):
    - learning_rate = 0.01
    - discount_factor (gamma) = 0.9
    - network_sync_rate = 1000   (número de passos para sincronizar as redes; ajustado para MountainCar)
    - replay_memory_size = 100000
    - mini_batch_size = 32
    - Rede neural: 1 camada oculta com 10 neurônios
    - Episódios = 1000 (pode ser ajustado)
    - max_steps = 200
    - epsilon_inicial = 1.0, epsilon_min = 0.01, epsilon_decay = 0.995

Requisitos:
    pip install gymnasium numpy matplotlib pandas seaborn torch
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sns.set(style="whitegrid", font_scale=1.1)

# Define a rede neural DQN
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Replay Memory
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)

# Deep Q-Learning Agent em PyTorch
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, memory_size=100000, network_sync_rate=1000, hidden_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.network_sync_rate = network_sync_rate
        self.step_count = 0

        self.policy_net = DQN(state_size, hidden_size, action_size)
        self.target_net = DQN(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values).item())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])
        
        # Previsão dos Q-values atuais
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        # Previsão dos Q-values do próximo estado usando a target network
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Atualiza epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

# Função de Treinamento
def train_dqn(env, agent, episodes=1000, max_steps=200):
    rewards_per_episode = []
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.step_count += 1
            if agent.step_count % agent.network_sync_rate == 0:
                agent.update_target_network()
            agent.replay()
            if done or truncated:
                break
        rewards_per_episode.append(total_reward)
        if episode % 10 == 0:
            print(f"Episódio {episode}/{episodes} - Recompensa: {total_reward:.2f}")
    return rewards_per_episode

def save_and_plot(rewards, prefix):
    df = pd.DataFrame({"episode": np.arange(1, len(rewards)+1), "reward": rewards})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{prefix}_rewards_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Dados salvos em {csv_filename}")
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="episode", y="reward", errorbar="sd", estimator="mean")
    df["moving_avg"] = df["reward"].rolling(window=10, min_periods=1).mean()
    plt.plot(df["episode"], df["moving_avg"], linestyle="--", label="Média móvel")
    plt.title(f"Curva de Aprendizado - {prefix}")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"{prefix}_learning_curve_{timestamp}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Gráfico salvo em {plot_filename}")
    return csv_filename, plot_filename

def main():
    parser = argparse.ArgumentParser(description="Treinamento Deep Q-Learning (PyTorch) para o ambiente MountainCar-v0")
    parser.add_argument("--runs", type=int, default=5, help="Número de execuções (N >= 5)")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episódios por execução")
    parser.add_argument("--max_steps", type=int, default=200, help="Número máximo de passos por episódio")
    args = parser.parse_args()
    
    env = gym.make("MountainCar-v0").env
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    all_rewards = []
    for run in range(1, args.runs+1):
        print(f"\nIniciando execução {run}/{args.runs}...")
        agent = DQNAgent(state_size, action_size, learning_rate=0.01, gamma=0.9,
                         epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                         batch_size=32, memory_size=100000, network_sync_rate=1000, hidden_size=10)
        rewards = train_dqn(env, agent, episodes=args.episodes, max_steps=args.max_steps)
        all_rewards.append(rewards)
        
        # Salva os pesos da rede neural para esta execução
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"mountaincar_deep_q_model_run{run}_{timestamp}.pt"
        agent.save(model_filename)
        print(f"Modelo salvo em {model_filename}")
        
        # Salva os dados da curva de aprendizado para esta execução
        save_and_plot(rewards, f"mountaincar_deep_q_run{run}")
    
    # Gera gráfico com a média entre execuções
    avg_rewards = np.mean(np.array(all_rewards), axis=0)
    df_avg = pd.DataFrame({"episode": np.arange(1, args.episodes+1), "reward": avg_rewards})
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_avg, x="episode", y="reward", errorbar="sd", estimator="mean", label="Média entre execuções")
    df_avg["moving_avg"] = df_avg["reward"].rolling(window=10, min_periods=1).mean()
    plt.plot(df_avg["episode"], df_avg["moving_avg"], linestyle="--", label="Média móvel")
    plt.title("Curva de Aprendizado - Deep Q-Learning (Média entre Execuções)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    overall_plot = f"mountaincar_deep_q_overall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(overall_plot)
    plt.close()
    print(f"Gráfico geral salvo em {overall_plot}")
    
    env.close()

if __name__ == "__main__":
    main()