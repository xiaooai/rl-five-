"""
测试改进后的五子棋环境
"""
import numpy as np
from game.environment import GomokuEnvironment
from game.factory import GomokuEnvironmentFactory, create_gomoku_env
from game.config import GomokuConfig


def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    env = GomokuEnvironment(board_size=9)
    obs, info = env.reset()
    
    print(f"观察空间形状: {obs.shape}")
    print(f"动作空间大小: {env.action_space.n}")
    print(f"初始信息: {info}")
    
    # 执行一些随机动作
    for i in range(5):
        valid_actions = env.get_valid_actions()
        valid_action_indices = np.where(valid_actions)[0]
        
        if len(valid_action_indices) > 0:
            action = np.random.choice(valid_action_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 结束={terminated}")
            
            if terminated:
                print("游戏结束!")
                break
    
    env.close()
    print("基本功能测试完成!\n")


def test_different_configs():
    """测试不同配置"""
    print("测试不同配置...")
    
    configs = ['standard', 'sparse', 'dense', 'small']
    
    for config_name in configs:
        print(f"测试配置: {config_name}")
        env = create_gomoku_env(config_name)
        obs, info = env.reset()
        
        print(f"  观察空间形状: {obs.shape}")
        print(f"  奖励类型: {env.reward_type}")
        print(f"  棋盘大小: {env.board_size}")
        
        # 执行一个动作
        valid_actions = env.get_valid_actions()
        if len(np.where(valid_actions)[0]) > 0:
            action = np.where(valid_actions)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  第一步奖励: {reward:.2f}")
        
        env.close()
    
    print("配置测试完成!\n")


def test_advanced_state():
    """测试高级状态表示"""
    print("测试高级状态表示...")
    
    # 标准状态
    env_standard = GomokuEnvironment(board_size=9, use_advanced_state=False)
    obs_standard, _ = env_standard.reset()
    print(f"标准状态形状: {obs_standard.shape}")
    
    # 高级状态
    env_advanced = GomokuEnvironment(board_size=9, use_advanced_state=True)
    obs_advanced, _ = env_advanced.reset()
    print(f"高级状态形状: {obs_advanced.shape}")
    
    # 比较状态内容
    print("状态通道含义:")
    print("  标准状态: [黑子, 白子, 当前玩家, 空位]")
    print("  高级状态: [黑子, 白子, 当前玩家, 空位, 位置信息, 历史信息]")
    
    env_standard.close()
    env_advanced.close()
    print("高级状态测试完成!\n")


def test_reward_types():
    """测试不同奖励类型"""
    print("测试不同奖励类型...")
    
    reward_types = ['standard', 'sparse', 'dense']
    
    for reward_type in reward_types:
        print(f"测试奖励类型: {reward_type}")
        env = GomokuEnvironment(board_size=9, reward_type=reward_type)
        obs, info = env.reset()
        
        total_reward = 0
        for i in range(10):
            valid_actions = env.get_valid_actions()
            valid_action_indices = np.where(valid_actions)[0]
            
            if len(valid_action_indices) > 0:
                action = np.random.choice(valid_action_indices)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated:
                    break
        
        print(f"  总奖励: {total_reward:.2f}")
        env.close()
    
    print("奖励类型测试完成!\n")


def test_game_logic():
    """测试游戏逻辑"""
    print("测试游戏逻辑...")
    
    env = GomokuEnvironment(board_size=9)
    obs, info = env.reset()
    
    # 测试无效移动
    invalid_action = 0  # 假设第一个位置
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    print(f"无效移动奖励: {reward}")
    
    # 重置并测试有效移动
    obs, info = env.reset()
    valid_actions = env.get_valid_actions()
    valid_action_indices = np.where(valid_actions)[0]
    
    if len(valid_action_indices) > 0:
        action = valid_action_indices[0]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"有效移动奖励: {reward}")
        print(f"游戏状态: {info['game_state']}")
    
    env.close()
    print("游戏逻辑测试完成!\n")


def test_rendering():
    """测试渲染功能"""
    print("测试渲染功能...")
    
    env = GomokuEnvironment(board_size=9, render_mode='human')
    obs, info = env.reset()
    
    print("初始棋盘:")
    env.render()
    
    # 执行几步
    for i in range(3):
        valid_actions = env.get_valid_actions()
        valid_action_indices = np.where(valid_actions)[0]
        
        if len(valid_action_indices) > 0:
            action = np.random.choice(valid_action_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\n第 {i+1} 步后:")
            env.render()
            
            if terminated:
                break
    
    env.close()
    print("渲染测试完成!\n")


def test_factory():
    """测试工厂类"""
    print("测试工厂类...")
    
    # 测试预定义配置
    configs = GomokuEnvironmentFactory.list_configs()
    print("可用配置:")
    for name, description in configs.items():
        print(f"  {name}: {description}")
    
    # 测试创建不同环境
    env1 = GomokuEnvironmentFactory.create_standard()
    env2 = GomokuEnvironmentFactory.create_sparse()
    env3 = GomokuEnvironmentFactory.create_dense()
    
    print(f"标准环境: {env1}")
    print(f"稀疏环境: {env2}")
    print(f"密集环境: {env3}")
    
    env1.close()
    env2.close()
    env3.close()
    print("工厂测试完成!\n")


def main():
    """主测试函数"""
    print("开始测试改进后的五子棋环境...")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_different_configs()
        test_advanced_state()
        test_reward_types()
        test_game_logic()
        test_rendering()
        test_factory()
        
        print("所有测试完成! ✅")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
