#!/usr/bin/env python3
"""
Deep Q-Learning para o ambiente MountainCar-v0 com otimizações para acelerar o treinamento.

Hiperparâmetros ajustados:
  - Rede neural: 2 camadas ocultas com 16 neurônios cada.
  - gamma = 0.99
  - epsilon_inicial = 1.0
  - epsilon_min = 0.01
  - epsilon_decay = 0.995
  - episodes = 1000 (ajustável)
  - max_steps = 200
  - batch_size = 16
  - memory_size = 5000
  - update_frequency = 8 (atualiza a rede a cada 8 passos)
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

sns.set(style="whitegrid", font_scale=1.1)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=16, memory_size=5000, update_frequency=8):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.update_frequency = update_frequency
        self.model = self._build_model()

    def _build_model(self):
        # Arquitetura simplificada: 2 camadas ocultas com 16 neurônios cada
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.state_size, dtype='float32'))
        model.add(Dense(16, activation='relu', dtype='float32'))
        model.add(Dense(self.action_size, activation='linear', dtype='float32'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state, dtype=np.float32).reshape(1, self.state_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in batch], dtype=np.float32)
        actions = np.array([s[1] for s in batch])
        rewards = np.array([s[2] for s in batch])
        next_states = np.array([s[3] for s in batch], dtype=np.float32)
        dones = np.array([s[4] for s in batch])
        
        q_next = self.model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.amax(q_next, axis=1) * (1 - dones)
        
        q_values = self.model.predict(states, verbose=0)
        for i in range(self.batch_size):
            q_values[i][actions[i]] = targets[i]
        
        self.model.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = tf.keras.models.load_model(name)
    
    def save(self, name):
        self.model.save(name)

def train_dqn(env, agent, episodes=1000, max_steps=200):
    rewards_list = []
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_state = np.array(next_state, dtype=np.float32)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # Atualiza a rede apenas a cada 'update_frequency' passos
            if step % agent.update_frequency == 0:
                agent.replay()
            if done or truncated:
                break
        rewards_list.append(total_reward)
        if episode % 10 == 0:
            print(f"Episódio {episode}/{episodes} - Recompensa: {total_reward}")
    return rewards_list

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
    parser = argparse.ArgumentParser(description="Treinamento Deep Q-Learning para o ambiente MountainCar-v0")
    parser.add_argument("--runs", type=int, default=5, help="Número de execuções (N >= 5)")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episódios por execução")
    parser.add_argument("--max_steps", type=int, default=200, help="Número máximo de passos por episódio")
    args = parser.parse_args()
    
    env = gym.make("MountainCar-v0").env
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    all_rewards = []  # Armazena recompensas de cada execução
    for run in range(1, args.runs+1):
        print(f"\nIniciando execução {run}/{args.runs}...")
        agent = DQNAgent(state_size, action_size, learning_rate=0.001, gamma=0.99,
                         epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                         batch_size=32, memory_size=5000, update_frequency=8)
        rewards = train_dqn(env, agent, episodes=args.episodes, max_steps=args.max_steps)
        all_rewards.append(rewards)
        
        # Salva os pesos da rede neural para esta execução
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"mountaincar_deep_q_model_run{run}_{timestamp}.h5"
        agent.save(model_filename)
        print(f"Modelo salvo em {model_filename}")
        
        # Salva os dados da curva de aprendizado para esta execução
        save_and_plot(rewards, f"mountaincar_deep_q_run{run}")
    
    # Gráfico geral com média entre execuções
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