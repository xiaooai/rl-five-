"""
PPO五子棋AI人机对战脚本
"""
import os
import argparse
import numpy as np
import torch
from game.environment import GomokuEnvironment
from ppo.agent import PPOAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO五子棋AI人机对战')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--board-size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    
    # 游戏参数
    parser.add_argument('--human-first', action='store_true', help='人类先手')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"使用设备: {device}")
    return device


def load_agent(model_path, board_size, hidden_dim, device):
    """加载智能体"""
    agent = PPOAgent(
        board_size=board_size,
        hidden_dim=hidden_dim,
        device=device
    )
    
    agent.load_model(model_path)
    return agent


def get_human_move(env):
    """获取人类玩家的移动"""
    while True:
        try:
            move = input("请输入您的移动 (格式: 行 列，例如: 7 7): ").strip()
            if move.lower() in ['quit', 'exit', 'q']:
                return None
            
            parts = move.split()
            if len(parts) != 2:
                print("请输入两个数字，用空格分隔")
                continue
            
            row, col = int(parts[0]), int(parts[1])
            
            if not env.board.is_valid_move(row, col):
                print("无效的移动，请重新输入")
                continue
            
            return row * env.board_size + col
            
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n游戏结束")
            return None


def get_ai_move(agent, env):
    """获取AI的移动"""
    state = env._get_observation()
    action_mask = env.get_action_mask()
    
    action, _, _ = agent.get_action(state, action_mask)
    
    row = action // env.board_size
    col = action % env.board_size
    
    return action, row, col


def display_board(env):
    """显示棋盘"""
    print("\n当前棋盘:")
    env.render()
    print()


def play_game(agent, env, human_first=True):
    """进行一局游戏"""
    print("="*50)
    print("五子棋人机对战")
    print("="*50)
    print("游戏规则:")
    print("- 黑子(●)先手，白子(○)后手")
    print("- 先连成5子者获胜")
    print("- 输入 'quit' 或 'q' 退出游戏")
    print("- 输入格式: 行 列 (例如: 7 7)")
    print("="*50)
    
    if human_first:
        print("您执黑子(●)，AI执白子(○)")
        human_player = 1
        ai_player = -1
    else:
        print("AI执黑子(●)，您执白子(○)")
        human_player = -1
        ai_player = 1
    
    print()
    
    # 重置环境
    state = env.reset()
    display_board(env)
    
    current_player = 1  # 黑子先手
    
    while True:
        if current_player == human_player:
            # 人类回合
            print(f"轮到您了 ({'黑子●' if human_player == 1 else '白子○'})")
            action = get_human_move(env)
            
            if action is None:
                print("游戏结束")
                return
            
            row = action // env.board_size
            col = action % env.board_size
            
        else:
            # AI回合
            print(f"AI思考中... ({'黑子●' if ai_player == 1 else '白子○'})")
            action, row, col = get_ai_move(agent, env)
            print(f"AI选择: ({row}, {col})")
        
        # 执行移动
        env.board.make_move(row, col)
        display_board(env)
        
        # 检查游戏状态
        winner = env.board.check_winner()
        
        if winner != 0:
            if winner == 1:
                if human_player == 1:
                    print("🎉 恭喜！您获胜了！")
                else:
                    print("😔 AI获胜了！")
            elif winner == -1:
                if human_player == -1:
                    print("🎉 恭喜！您获胜了！")
                else:
                    print("😔 AI获胜了！")
            else:
                print("🤝 平局！")
            
            break
        
        # 切换玩家
        current_player *= -1
    
    print("\n游戏结束！")


def main():
    """主函数"""
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建环境
    env = GomokuEnvironment(args.board_size)
    
    # 加载智能体
    print(f"加载模型: {args.model_path}")
    agent = load_agent(args.model_path, args.board_size, args.hidden_dim, device)
    
    try:
        # 开始游戏
        play_game(agent, env, args.human_first)
        
        # 询问是否继续
        while True:
            choice = input("\n是否继续游戏？(y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                play_game(agent, env, args.human_first)
            elif choice in ['n', 'no']:
                print("感谢游戏！再见！")
                break
            else:
                print("请输入 y 或 n")
    
    except KeyboardInterrupt:
        print("\n\n游戏结束，再见！")


if __name__ == '__main__':
    main()
