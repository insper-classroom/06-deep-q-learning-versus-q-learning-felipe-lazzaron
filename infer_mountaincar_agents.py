#!/usr/bin/env python3
"""
Avaliação dos agentes treinados (Q-Learning e Deep Q-Learning) no ambiente MountainCar-v0.

Este script:
  - Carrega a Q-table salva para o agente Q-Learning e o modelo salvo para o agente Deep Q-Learning.
  - Executa 100 episódios de inferência para cada agente (sem treinamento) e coleta as recompensas acumuladas.
  - Plota um gráfico comparativo das curvas de desempenho (reward por episódio) e exibe a média de recompensas.

Requisitos:
    pip install gymnasium numpy matplotlib pandas seaborn torch
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sns.set(style="whitegrid", font_scale=1.2)

# Função para criar bins para discretização do estado no MountainCar-v0
def create_bins(env):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    bins = np.round((env_high - env_low) * np.array([10, 100]), 0).astype(int) + 1
    return bins

# Função para discretizar um estado contínuo
def discretize_state(state, env, bins):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    ratios = (state - env_low) / (env_high - env_low)
    new_state = (ratios * (bins - 1)).astype(int)
    new_state = np.clip(new_state, 0, bins - 1)
    return tuple(new_state)

# Define a rede neural DQN (deve ser a mesma usada no treinamento Deep Q-Learning)
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Função de avaliação para o agente Q-Learning (tabular)
def evaluate_qlearning(env, q_table, bins, episodes=100):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        for step in range(200):
            disc_state = discretize_state(state, env, bins)
            action = int(np.argmax(q_table[disc_state]))
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = np.array(next_state, dtype=np.float32)
            if done or truncated:
                break
        rewards.append(total_reward)
    return rewards

# Função de avaliação para o agente Deep Q-Learning
def evaluate_deepq(env, model, episodes=100):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        for step in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = int(torch.argmax(q_values).item())
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = np.array(next_state, dtype=np.float32)
            if done or truncated:
                break
        rewards.append(total_reward)
    return rewards

def plot_inference(q_rewards, dqn_rewards, output_filename):
    episodes = np.arange(1, len(q_rewards)+1)
    plt.figure(figsize=(12,8))
    plt.plot(episodes, q_rewards, label="Q-Learning", marker="o")
    plt.plot(episodes, dqn_rewards, label="Deep Q-Learning", marker="s")
    plt.title("Desempenho em Inferência - MountainCar (100 episódios)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Gráfico de inferência salvo em {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Avaliação de Agentes treinados no MountainCar-v0")
    parser.add_argument("--q_table", type=str, required=True,
                        help="Caminho para o arquivo CSV da Q-table do agente Q-Learning (salvo como vetor flat)")
    parser.add_argument("--deep_model", type=str, required=True,
                        help="Caminho para o arquivo do modelo Deep Q-Learning (.pt)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Número de episódios para avaliação")
    parser.add_argument("--output", type=str, default="infer_mountaincar.png",
                        help="Nome do arquivo PNG para salvar o gráfico de inferência")
    args = parser.parse_args()
    
    # Cria o ambiente MountainCar
    env = gym.make("MountainCar-v0").env
    bins = create_bins(env)  # Ex: array([19,15])
    
    # Carrega a Q-table e a reestrutura para o formato correto: (bins[0], bins[1], action_size)
    q_table_flat = np.loadtxt(args.q_table, delimiter=",")
    action_size = env.action_space.n
    q_table = q_table_flat.reshape((bins[0], bins[1], action_size))
    
    # Carrega o modelo Deep Q-Learning
    state_size = env.observation_space.shape[0]
    deep_model = DQN(in_states=state_size, h1_nodes=10, out_actions=action_size)
    deep_model.load_state_dict(torch.load(args.deep_model))
    deep_model.eval()
    
    # Avaliação dos agentes
    print("Avaliando agente Q-Learning em inferência...")
    q_rewards = evaluate_qlearning(env, q_table, bins, episodes=args.episodes)
    print("Avaliando agente Deep Q-Learning em inferência...")
    dqn_rewards = evaluate_deepq(env, deep_model, episodes=args.episodes)
    
    # Plota o gráfico comparativo
    plot_inference(q_rewards, dqn_rewards, args.output)
    
    # Calcula e imprime as médias
    print(f"Q-Learning: Média de recompensa: {np.mean(q_rewards):.2f}")
    print(f"Deep Q-Learning: Média de recompensa: {np.mean(dqn_rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()