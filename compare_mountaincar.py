#!/usr/bin/env python3
"""
Script para comparar os resultados de Q-Learning e Deep Q-Learning no ambiente MountainCar-v0.

Os resultados estão salvos em arquivos CSV separados para cada execução.
Os arquivos de Q-Learning devem ter um prefixo (ex: "mountaincar_qlearning_")
e os arquivos de Deep Q-Learning devem ter um prefixo (ex: "mountaincar_deep_q_").

O script:
  - Carrega todos os CSVs para cada método.
  - Agrega os resultados (calculando a média dos rewards por episódio) usando o menor número de episódios encontrado.
  - Plota as curvas de aprendizado comparativas.
  - Adiciona uma linha horizontal indicando a meta de recompensa acumulada (ex: -110).

Uso:
    python3 compare_mountaincar.py --prefix_q "mountaincar_qlearning_" --prefix_dqn "mountaincar_deep_q_" --target -110 --output comparison_mountaincar.png --folder .
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set(style="whitegrid", font_scale=1.2)

def load_all_csvs(prefix, folder="."):
    """
    Carrega todos os arquivos CSV no 'folder' cujo nome começa com 'prefix'
    e retorna uma lista de DataFrames.
    """
    pattern = f"{folder}/{prefix}*.csv"
    files = glob.glob(pattern)
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)
    return df_list

def aggregate_results(df_list):
    """
    Agrega os resultados de uma lista de DataFrames.
    Se a coluna 'reward' não estiver presente:
      - Se o DataFrame tiver 2 colunas, renomeia-as para ["episode", "reward"].
      - Se tiver 1 coluna, assume que ela é 'reward' e cria uma coluna 'episode' automaticamente.
    Usa o menor número de episódios entre os DataFrames para garantir formato homogêneo.
    Retorna um DataFrame com a média dos rewards por episódio.
    """
    if not df_list:
        raise ValueError("Nenhum arquivo CSV encontrado.")
    
    # Determine o menor número de episódios entre os DataFrames
    n_episodes = min(df.shape[0] for df in df_list)
    
    rewards_all = []
    
    for df in df_list:
        # Se não houver a coluna "reward", tente renomear ou criar
        if "reward" not in df.columns:
            if df.shape[1] == 2:
                df.columns = ["episode", "reward"]
            elif df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "reward"})
                df.insert(0, "episode", np.arange(1, len(df)+1))
            else:
                raise ValueError("Número inesperado de colunas no CSV.")
        # Corte os DataFrames para ter apenas n_episodes linhas
        rewards_all.append(df["reward"].values[:n_episodes])
    
    avg_rewards = np.mean(np.array(rewards_all), axis=0)
    df_avg = pd.DataFrame({
        "episode": np.arange(1, n_episodes+1),
        "reward": avg_rewards
    })
    return df_avg

def plot_comparison(df_q, df_dqn, target_reward, output_filename):
    """
    Plota a curva de aprendizado comparativa entre Q-Learning e Deep Q-Learning.
    
    df_q: DataFrame agregado dos resultados do Q-Learning.
    df_dqn: DataFrame agregado dos resultados do Deep Q-Learning.
    target_reward: Valor da meta de recompensa acumulada (linha horizontal).
    output_filename: Nome do arquivo PNG para salvar o gráfico.
    """
    plt.figure(figsize=(12, 8))
    
    # Plota as curvas de aprendizado
    plt.plot(df_q["episode"], df_q["reward"], label="Q-Learning", marker="o")
    plt.plot(df_dqn["episode"], df_dqn["reward"], label="Deep Q-Learning", marker="s")
    
    # Adiciona médias móveis (janela de 10 episódios)
    df_q["moving_avg"] = df_q["reward"].rolling(window=10, min_periods=1).mean()
    df_dqn["moving_avg"] = df_dqn["reward"].rolling(window=10, min_periods=1).mean()
    plt.plot(df_q["episode"], df_q["moving_avg"], linestyle="--", label="Q-Learning (média móvel)")
    plt.plot(df_dqn["episode"], df_dqn["moving_avg"], linestyle="--", label="Deep Q-Learning (média móvel)")
    
    # Linha horizontal para a meta de recompensa
    plt.axhline(y=target_reward, color='red', linestyle='--', label=f"Meta: {target_reward}")
    
    plt.title("Comparação de Curva de Aprendizado - MountainCar")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Gráfico comparativo salvo em {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Comparação entre Q-Learning e Deep Q-Learning no ambiente MountainCar")
    parser.add_argument("--prefix_q", type=str, required=True,
                        help="Prefixo dos arquivos CSV do Q-Learning (ex: 'mountaincar_qlearning_')")
    parser.add_argument("--prefix_dqn", type=str, required=True,
                        help="Prefixo dos arquivos CSV do Deep Q-Learning (ex: 'mountaincar_deep_q_')")
    parser.add_argument("--target", type=float, default=-110,
                        help="Meta de recompensa acumulada (ex: -110)")
    parser.add_argument("--output", type=str, default="comparison_mountaincar.png",
                        help="Nome do arquivo PNG de saída para o gráfico")
    parser.add_argument("--folder", type=str, default=".", help="Pasta onde estão os CSVs")
    args = parser.parse_args()
    
    df_list_q = load_all_csvs(args.prefix_q, folder=args.folder)
    df_list_dqn = load_all_csvs(args.prefix_dqn, folder=args.folder)
    
    df_q = aggregate_results(df_list_q)
    df_dqn = aggregate_results(df_list_dqn)
    
    plot_comparison(df_q, df_dqn, args.target, args.output)

if __name__ == "__main__":
    main()