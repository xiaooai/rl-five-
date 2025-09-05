"""
基本功能测试脚本
"""
import numpy as np
import torch
from game.environment import GomokuEnvironment
from model.network import ActorCritic
from ppo.agent import PPOAgent


def test_environment():
    """测试游戏环境"""
    print("测试游戏环境...")
    env = GomokuEnvironment(15)
    
    # 测试重置
    state = env.reset()
    assert state.shape == (3, 15, 15), f"状态形状错误: {state.shape}"
    print("✓ 环境重置正常")
    
    # 测试有效动作
    valid_actions = env.get_valid_actions()
    assert len(valid_actions) == 225, f"有效动作数量错误: {len(valid_actions)}"
    print("✓ 有效动作获取正常")
    
    # 测试执行动作
    action = 112  # 中心位置
    state, reward, done, info = env.step(action)
    assert not done, "第一步不应该结束游戏"
    print("✓ 动作执行正常")
    
    print("游戏环境测试通过！\n")


def test_network():
    """测试神经网络"""
    print("测试神经网络...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = ActorCritic(15, 512).to(device)
    
    # 测试前向传播
    batch_size = 4
    state = torch.randn(batch_size, 3, 15, 15).to(device)
    action_probs, action_log_probs, value = network(state)
    
    assert action_probs.shape == (batch_size, 225), f"动作概率形状错误: {action_probs.shape}"
    assert action_log_probs.shape == (batch_size, 225), f"动作对数概率形状错误: {action_log_probs.shape}"
    assert value.shape == (batch_size, 1), f"价值形状错误: {value.shape}"
    print("✓ 网络前向传播正常")
    
    # 测试动作选择
    action_mask = torch.ones(batch_size, 225).to(device)
    action, log_prob, value = network.get_action(state, action_mask)
    
    assert action.shape == (batch_size, 1), f"动作形状错误: {action.shape}"
    assert log_prob.shape == (batch_size, 1), f"对数概率形状错误: {log_prob.shape}"
    assert value.shape == (batch_size, 1), f"价值形状错误: {value.shape}"
    print("✓ 动作选择正常")
    
    print("神经网络测试通过！\n")


def test_agent():
    """测试PPO智能体"""
    print("测试PPO智能体...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(15, 512, device=device)
    
    # 测试动作选择
    env = GomokuEnvironment(15)
    state = env.reset()
    action_mask = env.get_action_mask()
    
    action, log_prob, value = agent.get_action(state, action_mask)
    assert isinstance(action, int), f"动作类型错误: {type(action)}"
    assert isinstance(log_prob, float), f"对数概率类型错误: {type(log_prob)}"
    assert isinstance(value, float), f"价值类型错误: {type(value)}"
    print("✓ 智能体动作选择正常")
    
    # 测试经验收集
    rollout_stats = agent.collect_rollout(env, 100)
    assert 'episode_rewards' in rollout_stats, "经验收集缺少episode_rewards"
    assert 'episode_lengths' in rollout_stats, "经验收集缺少episode_lengths"
    print("✓ 经验收集正常")
    
    # 测试策略更新
    if len(agent.rollout_buffer) > 0:
        update_stats = agent.update(1, 32)
        assert 'policy_loss' in update_stats, "策略更新缺少policy_loss"
        print("✓ 策略更新正常")
    
    print("PPO智能体测试通过！\n")


def main():
    """主测试函数"""
    print("开始基本功能测试...\n")
    
    try:
        test_environment()
        test_network()
        test_agent()
        
        print("🎉 所有测试通过！项目基本功能正常。")
        print("\n可以开始训练了：")
        print("python train.py")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
