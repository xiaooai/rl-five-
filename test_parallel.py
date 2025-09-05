"""
测试并行环境实现
"""
import numpy as np
import torch
import time
from game.environment import GomokuEnvironment
from game.parallel_env import ParallelEnvironment, VectorizedEnvironment
from ppo.agent import PPOAgent


def test_single_environment():
    """测试单个环境"""
    print("测试单个环境...")
    
    env = GomokuEnvironment(15)
    agent = PPOAgent(board_size=15, device='cpu')
    
    start_time = time.time()
    
    # 收集经验
    rollout_stats = agent.collect_rollout(env, 100)
    
    end_time = time.time()
    
    print(f"单个环境 - 时间: {end_time - start_time:.2f}秒")
    print(f"回合奖励: {rollout_stats['episode_rewards']}")
    print(f"回合长度: {rollout_stats['episode_lengths']}")
    print(f"总步数: {rollout_stats['total_steps']}")
    print()


def test_parallel_environment():
    """测试并行环境"""
    print("测试并行环境...")
    
    num_envs = 4
    env = ParallelEnvironment(num_envs, 15, 'cpu')
    agent = PPOAgent(board_size=15, device='cpu')
    
    start_time = time.time()
    
    # 收集经验
    rollout_stats = agent.collect_rollout(env, 100)
    
    end_time = time.time()
    
    print(f"并行环境 ({num_envs}个) - 时间: {end_time - start_time:.2f}秒")
    print(f"回合奖励: {rollout_stats['episode_rewards']}")
    print(f"回合长度: {rollout_stats['episode_lengths']}")
    print(f"总步数: {rollout_stats['total_steps']}")
    print()


def test_vectorized_environment():
    """测试向量化环境"""
    print("测试向量化环境...")
    
    num_envs = 4
    env = VectorizedEnvironment(num_envs, 15, 'cpu')
    agent = PPOAgent(board_size=15, device='cpu')
    
    start_time = time.time()
    
    # 收集经验
    rollout_stats = agent.collect_rollout(env, 100)
    
    end_time = time.time()
    
    print(f"向量化环境 ({num_envs}个) - 时间: {end_time - start_time:.2f}秒")
    print(f"回合奖励: {rollout_stats['episode_rewards']}")
    print(f"回合长度: {rollout_stats['episode_lengths']}")
    print(f"总步数: {rollout_stats['total_steps']}")
    print()


def test_environment_consistency():
    """测试环境一致性"""
    print("测试环境一致性...")
    
    # 创建单个环境
    single_env = GomokuEnvironment(15)
    
    # 创建并行环境
    parallel_env = ParallelEnvironment(1, 15, 'cpu')
    
    # 重置环境
    single_state = single_env.reset()
    parallel_states = parallel_env.reset()
    
    print(f"单个环境状态形状: {single_state.shape}")
    print(f"并行环境状态形状: {parallel_states.shape}")
    
    # 检查状态是否相同
    if np.array_equal(single_state, parallel_states[0]):
        print("✓ 环境状态一致")
    else:
        print("✗ 环境状态不一致")
    
    # 测试动作掩码
    single_mask = single_env.get_action_mask()
    parallel_masks = parallel_env.get_action_masks()
    
    if np.array_equal(single_mask, parallel_masks[0]):
        print("✓ 动作掩码一致")
    else:
        print("✗ 动作掩码不一致")
    
    print()


def test_batch_actions():
    """测试批量动作选择"""
    print("测试批量动作选择...")
    
    num_envs = 4
    env = ParallelEnvironment(num_envs, 15, 'cpu')
    agent = PPOAgent(board_size=15, device='cpu')
    
    # 重置环境
    states = env.reset()
    action_masks = env.get_action_masks()
    
    # 批量选择动作
    actions, log_probs, values = agent.get_actions_batch(states, action_masks)
    
    print(f"状态形状: {states.shape}")
    print(f"动作掩码形状: {action_masks.shape}")
    print(f"动作形状: {actions.shape}")
    print(f"对数概率形状: {log_probs.shape}")
    print(f"价值形状: {values.shape}")
    
    # 检查动作是否有效
    for i in range(num_envs):
        if action_masks[i][actions[i]] == 1.0:
            print(f"✓ 环境 {i} 动作有效")
        else:
            print(f"✗ 环境 {i} 动作无效")
    
    print()


def main():
    """主测试函数"""
    print("开始测试并行环境实现...")
    print("=" * 50)
    
    # 测试环境一致性
    test_environment_consistency()
    
    # 测试批量动作选择
    test_batch_actions()
    
    # 性能测试
    print("性能测试:")
    print("-" * 30)
    
    test_single_environment()
    test_parallel_environment()
    test_vectorized_environment()
    
    print("测试完成!")


if __name__ == '__main__':
    main()
