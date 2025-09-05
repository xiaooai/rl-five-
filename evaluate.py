"""
PPO五子棋AI评估脚本
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from game.environment import GomokuEnvironment
from ppo.agent import PPOAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO五子棋AI评估')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--board-size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    
    # 评估参数
    parser.add_argument('--num-games', type=int, default=100, help='评估游戏数量')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./evaluation', help='输出目录')
    parser.add_argument('--save-games', action='store_true', help='保存游戏记录')
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"使用设备: {device}")
    return device


def create_directories(output_dir):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)


def load_agent(model_path, board_size, hidden_dim, device):
    """加载智能体"""
    agent = PPOAgent(
        board_size=board_size,
        hidden_dim=hidden_dim,
        device=device
    )
    
    agent.load_model(model_path)
    return agent


def play_game(agent, env, save_game=False):
    """进行一局游戏"""
    state = env.reset()
    game_record = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'winner': 0,
        'total_reward': 0,
        'game_length': 0
    }
    
    done = False
    while not done:
        # 获取动作掩码
        action_mask = env.get_action_mask()
        
        # 选择动作
        action, log_prob, value = agent.get_action(state, action_mask)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 记录游戏
        if save_game:
            game_record['states'].append(state.copy())
            game_record['actions'].append(action)
            game_record['rewards'].append(reward)
            game_record['dones'].append(done)
        
        game_record['total_reward'] += reward
        game_record['game_length'] += 1
        
        state = next_state
        
        if done:
            game_record['winner'] = info.get('winner', 0)
    
    return game_record


def evaluate_agent(agent, env, num_games, save_games=False):
    """评估智能体"""
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_rewards': [],
        'game_lengths': [],
        'win_rates': [],
        'games': []
    }
    
    print(f"开始评估，共 {num_games} 局游戏...")
    
    for game_idx in tqdm(range(num_games), desc="评估进度"):
        game_record = play_game(agent, env, save_games)
        
        # 统计结果
        if game_record['winner'] == 1:  # 黑子获胜
            results['wins'] += 1
        elif game_record['winner'] == -1:  # 白子获胜
            results['losses'] += 1
        else:  # 平局
            results['draws'] += 1
        
        results['total_rewards'].append(game_record['total_reward'])
        results['game_lengths'].append(game_record['game_length'])
        
        # 计算当前胜率
        current_win_rate = results['wins'] / (game_idx + 1)
        results['win_rates'].append(current_win_rate)
        
        if save_games:
            results['games'].append(game_record)
    
    return results


def analyze_results(results):
    """分析评估结果"""
    analysis = {}
    
    # 基本统计
    analysis['total_games'] = len(results['total_rewards'])
    analysis['wins'] = results['wins']
    analysis['losses'] = results['losses']
    analysis['draws'] = results['draws']
    analysis['win_rate'] = results['wins'] / analysis['total_games']
    analysis['loss_rate'] = results['losses'] / analysis['total_games']
    analysis['draw_rate'] = results['draws'] / analysis['total_games']
    
    # 奖励统计
    analysis['avg_reward'] = np.mean(results['total_rewards'])
    analysis['std_reward'] = np.std(results['total_rewards'])
    analysis['min_reward'] = np.min(results['total_rewards'])
    analysis['max_reward'] = np.max(results['total_rewards'])
    
    # 游戏长度统计
    analysis['avg_game_length'] = np.mean(results['game_lengths'])
    analysis['std_game_length'] = np.std(results['game_lengths'])
    analysis['min_game_length'] = np.min(results['game_lengths'])
    analysis['max_game_length'] = np.max(results['game_lengths'])
    
    return analysis


def plot_evaluation_results(results, analysis, save_path):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PPO五子棋AI评估结果', fontsize=16)
    
    # 胜率曲线
    axes[0, 0].plot(results['win_rates'])
    axes[0, 0].set_title('胜率变化')
    axes[0, 0].set_xlabel('游戏数')
    axes[0, 0].set_ylabel('胜率')
    axes[0, 0].grid(True)
    
    # 奖励分布
    axes[0, 1].hist(results['total_rewards'], bins=20, alpha=0.7)
    axes[0, 1].set_title('奖励分布')
    axes[0, 1].set_xlabel('总奖励')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].axvline(analysis['avg_reward'], color='red', linestyle='--', 
                      label=f'平均: {analysis["avg_reward"]:.2f}')
    axes[0, 1].legend()
    
    # 游戏长度分布
    axes[1, 0].hist(results['game_lengths'], bins=20, alpha=0.7)
    axes[1, 0].set_title('游戏长度分布')
    axes[1, 0].set_xlabel('游戏长度')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].axvline(analysis['avg_game_length'], color='red', linestyle='--',
                      label=f'平均: {analysis["avg_game_length"]:.1f}')
    axes[1, 0].legend()
    
    # 结果饼图
    labels = ['获胜', '失败', '平局']
    sizes = [analysis['wins'], analysis['losses'], analysis['draws']]
    colors = ['green', 'red', 'gray']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('游戏结果分布')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_results(results, analysis, output_dir):
    """保存评估结果"""
    # 保存分析结果
    analysis_path = os.path.join(output_dir, 'evaluation_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # 保存详细结果
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    # 移除游戏记录以减小文件大小
    results_to_save = {k: v for k, v in results.items() if k != 'games'}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # 保存游戏记录（如果存在）
    if results['games']:
        games_path = os.path.join(output_dir, 'game_records.json')
        with open(games_path, 'w', encoding='utf-8') as f:
            json.dump(results['games'], f, indent=2, ensure_ascii=False)


def print_evaluation_summary(analysis):
    """打印评估摘要"""
    print("\n" + "="*50)
    print("评估结果摘要")
    print("="*50)
    print(f"总游戏数: {analysis['total_games']}")
    print(f"获胜: {analysis['wins']} ({analysis['win_rate']:.1%})")
    print(f"失败: {analysis['losses']} ({analysis['loss_rate']:.1%})")
    print(f"平局: {analysis['draws']} ({analysis['draw_rate']:.1%})")
    print()
    print(f"平均奖励: {analysis['avg_reward']:.2f} ± {analysis['std_reward']:.2f}")
    print(f"奖励范围: [{analysis['min_reward']:.2f}, {analysis['max_reward']:.2f}]")
    print()
    print(f"平均游戏长度: {analysis['avg_game_length']:.1f} ± {analysis['std_game_length']:.1f}")
    print(f"游戏长度范围: [{analysis['min_game_length']}, {analysis['max_game_length']}]")
    print("="*50)


def main():
    """主评估函数"""
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建输出目录
    create_directories(args.output_dir)
    
    # 创建环境
    env = GomokuEnvironment(args.board_size)
    
    # 加载智能体
    print(f"加载模型: {args.model_path}")
    agent = load_agent(args.model_path, args.board_size, args.hidden_dim, device)
    
    # 评估智能体
    results = evaluate_agent(agent, env, args.num_games, args.save_games)
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 打印摘要
    print_evaluation_summary(analysis)
    
    # 保存结果
    save_evaluation_results(results, analysis, args.output_dir)
    
    # 绘制图表
    plot_path = os.path.join(args.output_dir, 'evaluation_results.png')
    plot_evaluation_results(results, analysis, plot_path)
    
    print(f"\n评估结果已保存到: {args.output_dir}")
    print(f"图表已保存到: {plot_path}")


if __name__ == '__main__':
    main()
