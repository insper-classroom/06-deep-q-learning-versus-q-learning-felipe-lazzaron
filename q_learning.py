#!/usr/bin/env python3
"""
Treinamento do agente Q-Learning para o ambiente MountainCar-v0.

Atividades:
- Treinar o agente usando Q-Learning para o ambiente MountainCar-v0 utilizando os melhores hiperparâmetros possíveis.
- Executar o treinamento N vezes (N >= 5) e coletar os dados para criar a curva de aprendizado.
- Armazenar os pesos da Q-table para cada execução.

O ambiente MountainCar-v0 possui estados contínuos (posição e velocidade). Para utilizar Q-Learning,
discretizamos o espaço de estados. A discretização é feita conforme:
  num_states = round((env.observation_space.high - env.observation_space.low) * [10, 100]) + 1
Para o MountainCar-v0, isso resulta em uma grade com dimensões (19, 15), e o número de ações é 3.

Hiperparâmetros padrão (sugeridos, ajuste conforme necessário):
  - alpha = 0.1
  - gamma = 0.99
  - epsilon inicial = 1.0
  - epsilon_min = 0.01
  - epsilon_decay = 0.995
  - episódios = 10000
  - max_steps = 200

Os resultados (recompensa acumulada por episódio) serão salvos em CSV e as Q-tables serão armazenadas em arquivos CSV com nomes únicos.
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

def discretize_state(state, env, bins):
    """Discretiza um estado contínuo usando os bins fornecidos."""
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    ratios = (state - env_low) / (env_high - env_low)
    new_state = (ratios * (bins - 1)).astype(int)
    new_state = np.clip(new_state, 0, bins - 1)
    return tuple(new_state)

def create_bins(env):
    """Cria os bins para discretizar o estado do MountainCar-v0.
       Conforme a fórmula: num_states = round((high - low) * [10, 100]) + 1.
       Resulta em bins para posição e velocidade."""
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    bins = np.round((env_high - env_low) * np.array([10, 100]), 0).astype(int) + 1
    return bins

class QLearningAgent:
    def __init__(self, env, bins, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.bins = bins  # bins para discretização (ex.: [19,15])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: dimensões = (num_bins_pos, num_bins_vel, num_actions)
        self.q_table = np.zeros((bins[0], bins[1], env.action_space.n))
    
    def select_action(self, state):
        """Seleciona uma ação usando a política epsilon-greedy."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        """Atualiza a Q-table usando a equação de Bellman."""
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next * (1 - int(done)) - self.q_table[state][action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def train_agent(env, agent, episodes, max_steps):
    rewards_per_episode = []
    for episode in range(episodes):
        state_cont, _ = env.reset()
        state = discretize_state(state_cont, env, agent.bins)
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state_cont, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_cont, env, agent.bins)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        if (episode+1) % 100 == 0:
            print(f"Episódio {episode+1}/{episodes} - Recompensa: {total_reward}")
    return rewards_per_episode

def save_csv(df, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Dados salvos em {filename}")
    return filename

def plot_learning_curve(rewards, prefix):
    df = pd.DataFrame({"episode": np.arange(1, len(rewards)+1), "reward": rewards})
    plt.figure(figsize=(12, 8))
    plt.plot(df["episode"], df["reward"], label="Recompensa")
    df["moving_avg"] = df["reward"].rolling(window=10, min_periods=1).mean()
    plt.plot(df["episode"], df["moving_avg"], linestyle="--", label="Média móvel")
    plt.title(f"Curva de Aprendizado - {prefix}")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"{prefix}_learning_curve.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Gráfico salvo em {plot_filename}")
    return plot_filename

def main():
    parser = argparse.ArgumentParser(description="Treinamento Q-Learning para o ambiente MountainCar-v0")
    parser.add_argument("--runs", type=int, default=5, help="Número de execuções (N >= 5)")
    parser.add_argument("--episodes", type=int, default=10000, help="Número de episódios por execução")
    parser.add_argument("--max_steps", type=int, default=200, help="Número máximo de passos por episódio")
    args = parser.parse_args()
    
    env = gym.make("MountainCar-v0").env
    bins = create_bins(env)  # Exemplo: array([19,15])
    print(f"Bins para discretização: {bins}")
    
    all_rewards = []  # Para armazenar as curvas de aprendizado de cada execução
    for run in range(1, args.runs + 1):
        print(f"\nIniciando execução {run}/{args.runs}...")
        agent = QLearningAgent(env, bins, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
        rewards = train_agent(env, agent, episodes=args.episodes, max_steps=args.max_steps)
        all_rewards.append(rewards)
        # Salva a Q-table desta execução
        qtable_filename = f"mountaincar_qlearning_qtable_run{run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        np.savetxt(qtable_filename, agent.q_table.reshape(-1), delimiter=",")
        print(f"Q-table da execução {run} salva em {qtable_filename}")
        # Salva os dados da curva de aprendizado desta execução
        df = pd.DataFrame({"episode": np.arange(1, args.episodes+1), "reward": rewards})
        csv_filename = save_csv(df, f"mountaincar_qlearning_rewards_run{run}")
        plot_learning_curve(rewards, f"mountaincar_qlearning_run{run}")
    
    # Opcional: calcular a média da curva de aprendizado entre as execuções
    avg_rewards = np.mean(np.array(all_rewards), axis=0)
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, args.episodes+1), avg_rewards, label="Média entre execuções")
    moving_avg = pd.Series(avg_rewards).rolling(window=10, min_periods=1).mean()
    plt.plot(np.arange(1, args.episodes+1), moving_avg, linestyle="--", label="Média móvel")
    plt.title("Curva de Aprendizado - Média entre Execuções (Q-Learning, MountainCar)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    overall_plot = f"mountaincar_qlearning_overall.png"
    plt.savefig(overall_plot)
    plt.close()
    print(f"Gráfico geral salvo em {overall_plot}")
    
    env.close()

if __name__ == "__main__":
    main()