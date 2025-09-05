"""
测试恢复训练功能
"""
import os
import torch
import numpy as np
from ppo.agent import PPOAgent
from game.environment import GomokuEnvironment


def test_save_and_load():
    """测试保存和加载模型"""
    print("测试保存和加载模型...")
    
    # 创建智能体
    agent = PPOAgent(board_size=15, device='cpu')
    
    # 添加一些模拟的训练统计
    agent.training_stats = {
        'episode_rewards': [10.0, 15.0, 20.0],
        'episode_lengths': [50, 60, 70],
        'win_rates': [0.3, 0.4, 0.5],
        'policy_loss': [0.1, 0.08, 0.06]
    }
    
    # 保存模型
    test_model_path = './test_model.pth'
    agent.save_model(test_model_path)
    print(f"模型已保存到: {test_model_path}")
    
    # 创建新的智能体并加载模型
    new_agent = PPOAgent(board_size=15, device='cpu')
    new_agent.load_model(test_model_path)
    
    # 验证训练统计是否正确加载
    loaded_stats = new_agent.get_training_stats()
    print(f"加载的训练统计: {loaded_stats}")
    
    # 验证模型参数是否相同
    original_params = list(agent.actor_critic.parameters())
    loaded_params = list(new_agent.actor_critic.parameters())
    
    params_match = True
    for orig, loaded in zip(original_params, loaded_params):
        if not torch.equal(orig, loaded):
            params_match = False
            break
    
    if params_match:
        print("✓ 模型参数加载正确")
    else:
        print("✗ 模型参数加载错误")
    
    # 清理测试文件
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print("测试文件已清理")
    
    return params_match


def test_checkpoint_management():
    """测试检查点管理功能"""
    print("\n测试检查点管理功能...")
    
    # 创建测试目录
    test_dir = './test_checkpoints'
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 创建一些测试检查点文件
        test_files = [
            'ppo_gomoku_episode_100.pth',
            'ppo_gomoku_episode_200.pth',
            'ppo_gomoku_episode_300.pth',
            'ppo_gomoku_episode_400.pth',
            'ppo_gomoku_episode_500.pth'
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test checkpoint")
        
        print(f"创建了 {len(test_files)} 个测试检查点")
        
        # 导入清理函数
        from train import find_latest_checkpoint, cleanup_old_checkpoints
        
        # 测试查找最新检查点
        latest = find_latest_checkpoint(test_dir)
        if latest and 'episode_500' in latest:
            print("✓ 正确找到最新检查点")
        else:
            print("✗ 未找到最新检查点")
        
        # 测试清理旧检查点
        cleanup_old_checkpoints(test_dir, keep_last=2)
        
        remaining_files = [f for f in os.listdir(test_dir) if f.endswith('.pth')]
        print(f"清理后剩余文件: {remaining_files}")
        
        if len(remaining_files) == 2:
            print("✓ 检查点清理正确")
        else:
            print("✗ 检查点清理错误")
        
    finally:
        # 清理测试目录
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("测试目录已清理")


def test_environment_compatibility():
    """测试环境兼容性"""
    print("\n测试环境兼容性...")
    
    # 创建环境
    env = GomokuEnvironment(15)
    
    # 创建智能体
    agent = PPOAgent(board_size=15, device='cpu')
    
    # 测试收集经验
    try:
        rollout_stats = agent.collect_rollout(env, 10)
        print(f"✓ 成功收集经验: {rollout_stats}")
    except Exception as e:
        print(f"✗ 收集经验失败: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("开始测试恢复训练功能...")
    print("=" * 50)
    
    # 测试保存和加载
    save_load_ok = test_save_and_load()
    
    # 测试检查点管理
    test_checkpoint_management()
    
    # 测试环境兼容性
    env_ok = test_environment_compatibility()
    
    print("\n测试结果:")
    print("=" * 50)
    print(f"保存/加载功能: {'✓ 通过' if save_load_ok else '✗ 失败'}")
    print(f"环境兼容性: {'✓ 通过' if env_ok else '✗ 失败'}")
    
    if save_load_ok and env_ok:
        print("\n所有测试通过! 恢复训练功能正常工作。")
    else:
        print("\n部分测试失败，请检查实现。")


if __name__ == "__main__":
    main()
