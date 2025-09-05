"""
PPO五子棋AI训练脚本
"""
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from game.environment import GomokuEnvironment
from game.parallel_env import ParallelEnvironment, VectorizedEnvironment
from ppo.agent import PPOAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO五子棋AI训练')
    
    # 环境参数
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    
    # 训练参数
    parser.add_argument('--num-episodes', type=int, default=10000, help='训练轮数')
    parser.add_argument('--num-steps', type=int, default=4096, help='每轮收集步数')
    parser.add_argument('--num-epochs', type=int, default=4, help='每次更新的轮数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批次大小')
    
    # 网络参数
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    
    # PPO参数
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--lambda-gae', type=float, default=0.95, help='GAE参数')
    parser.add_argument('--clip-ratio', type=float, default=0.2, help='PPO裁剪比例')
    parser.add_argument('--value-coef', type=float, default=0.5, help='价值损失系数')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='熵损失系数')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='梯度裁剪阈值')
    
    # 并行环境参数
    parser.add_argument('--num-envs', type=int, default=16, help='并行环境数量')
    parser.add_argument('--use-vectorized', action='store_true', help='使用向量化环境')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--save-interval', type=int, default=100, help='保存间隔')
    parser.add_argument('--eval-interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
    parser.add_argument('--save-dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"使用设备: {device}")
    return device


def create_directories(save_dir, log_dir):
    """创建必要的目录"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def evaluate_agent(agent, env, num_games=10):
    """评估智能体"""
    wins = 0
    total_rewards = []
    
    # 检查是否为并行环境
    if hasattr(env, 'num_envs'):
        # 使用单个环境进行评估
        eval_env = GomokuEnvironment(env.board_size)
    else:
        eval_env = env
    
    for _ in range(num_games):
        state = eval_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_mask = eval_env.get_action_mask()
            action, _, _ = agent.get_action(state, action_mask)
            state, reward, done, _ = eval_env.step(action)
            total_reward += reward
            
            if done and reward > 0:  # 获胜
                wins += 1
        
        total_rewards.append(total_reward)
    
    win_rate = wins / num_games
    avg_reward = np.mean(total_rewards)
    
    return win_rate, avg_reward


def plot_training_curves(stats, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PPO训练曲线', fontsize=16)
    
    # 奖励曲线
    if 'episode_rewards' in stats:
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
    
    # 策略损失
    if 'policy_loss' in stats:
        axes[0, 1].plot(stats['policy_loss'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
    
    # 价值损失
    if 'value_loss' in stats:
        axes[0, 2].plot(stats['value_loss'])
        axes[0, 2].set_title('Value Loss')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Loss')
    
    # 熵损失
    if 'entropy_loss' in stats:
        axes[1, 0].plot(stats['entropy_loss'])
        axes[1, 0].set_title('Entropy Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
    
    # KL散度
    if 'kl_divergence' in stats:
        axes[1, 1].plot(stats['kl_divergence'])
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('KL Divergence')
    
    # 裁剪比例
    if 'clip_fraction' in stats:
        axes[1, 2].plot(stats['clip_fraction'])
        axes[1, 2].set_title('Clip Fraction')
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Clip Fraction')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建目录
    create_directories(args.save_dir, args.log_dir)
    
    # 创建环境
    if args.num_envs > 1:
        if args.use_vectorized:
            env = VectorizedEnvironment(args.num_envs, args.board_size, device)
        else:
            env = ParallelEnvironment(args.num_envs, args.board_size, device)
        print(f"使用并行环境: {args.num_envs} 个环境")
    else:
        env = GomokuEnvironment(args.board_size)
        print("使用单个环境")
    
    # 创建智能体
    agent = PPOAgent(
        board_size=args.board_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device
    )
    
    # 恢复训练
    start_episode = 0
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        agent.load_model(args.resume)
        start_episode = len(agent.get_training_stats().get('episode_rewards', []))
    
    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'win_rates': [],
        'avg_rewards': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': [],
        'total_loss': [],
        'kl_divergence': [],
        'clip_fraction': []
    }
    
    print("开始训练...")
    print(f"训练参数: {vars(args)}")
    
    # 训练循环
    for episode in tqdm(range(start_episode, args.num_episodes), desc="训练进度"):
        # 收集经验
        rollout_stats = agent.collect_rollout(env, args.num_steps)
        
        # 更新策略
        update_stats = agent.update(args.num_epochs, args.batch_size)
        
        # 记录统计信息
        if rollout_stats['episode_rewards']:
            training_stats['episode_rewards'].extend(rollout_stats['episode_rewards'])
            training_stats['episode_lengths'].extend(rollout_stats['episode_lengths'])
        
        if update_stats:
            for key, value in update_stats.items():
                training_stats[key].append(value)
        
        # 评估
        if episode % args.eval_interval == 0 and episode > 0:
            win_rate, avg_reward = evaluate_agent(agent, env)
            training_stats['win_rates'].append(win_rate)
            training_stats['avg_rewards'].append(avg_reward)
            
            print(f"Episode {episode}: Win Rate = {win_rate:.3f}, Avg Reward = {avg_reward:.3f}")
        
        # 日志输出
        if episode % args.log_interval == 0 and episode > 0:
            recent_rewards = training_stats['episode_rewards'][-10:]
            avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            print(f"Episode {episode}: Avg Recent Reward = {avg_recent_reward:.3f}")
            
            if update_stats:
                print(f"  Policy Loss: {update_stats.get('policy_loss', 0):.4f}")
                print(f"  Value Loss: {update_stats.get('value_loss', 0):.4f}")
                print(f"  Entropy Loss: {update_stats.get('entropy_loss', 0):.4f}")
                print(f"  KL Divergence: {update_stats.get('kl_divergence', 0):.4f}")
        
        # 保存模型
        if episode % args.save_interval == 0 and episode > 0:
            model_path = os.path.join(args.save_dir, f'ppo_gomoku_episode_{episode}.pth')
            agent.save_model(model_path)
            print(f"模型已保存到: {model_path}")
            
            # 保存训练统计
            stats_path = os.path.join(args.log_dir, f'training_stats_episode_{episode}.json')
            with open(stats_path, 'w') as f:
                json.dump(convert_numpy_types(training_stats), f, indent=2)
            
            # 绘制训练曲线
            plot_path = os.path.join(args.log_dir, f'training_curves_episode_{episode}.png')
            plot_training_curves(training_stats, plot_path)
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'ppo_gomoku_final.pth')
    agent.save_model(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 保存最终统计
    final_stats_path = os.path.join(args.log_dir, 'final_training_stats.json')
    with open(final_stats_path, 'w') as f:
        json.dump(convert_numpy_types(training_stats), f, indent=2)
    
    # 绘制最终训练曲线
    final_plot_path = os.path.join(args.log_dir, 'final_training_curves.png')
    plot_training_curves(training_stats, final_plot_path)
    
    print("训练完成!")


if __name__ == '__main__':
    main()
