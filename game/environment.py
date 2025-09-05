"""
五子棋游戏环境
"""
import gymnasium as gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any
from .board import Board


class GomokuEnvironment(gym.Env):
    """五子棋游戏环境"""
    
    def __init__(self, board_size: int = 15):
        """
        初始化环境
        
        Args:
            board_size: 棋盘大小
        """
        super().__init__()
        
        self.board_size = board_size
        self.board = Board(board_size)
        
        # 动作空间：所有可能的落子位置
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 观察空间：编码后的棋盘状态
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(3, board_size, board_size), 
            dtype=np.float32
        )
        
        # 游戏统计
        self.episode_reward = 0
        self.episode_length = 0
        
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观察状态
        """
        self.board.reset()
        self.episode_reward = 0
        self.episode_length = 0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作索引
            
        Returns:
            (观察, 奖励, 是否结束, 信息)
        """
        # 将动作索引转换为坐标
        row = action // self.board_size
        col = action % self.board_size
        
        # 检查动作是否有效
        if not self.board.is_valid_move(row, col):
            # 无效动作，给予负奖励
            reward = -10.0
            done = True
            info = {"invalid_move": True}
            return self._get_observation(), reward, done, info
        
        # 执行落子
        self.board.make_move(row, col)
        self.episode_length += 1
        
        # 检查游戏状态
        winner = self.board.check_winner()
        
        if winner != 0:
            # 游戏结束
            done = True
            if winner == 1:  # 黑子获胜
                reward = 100.0
            elif winner == -1:  # 白子获胜
                reward = -100.0
            else:  # 平局
                reward = 0.0
        else:
            # 游戏继续
            done = False
            # 给予小的正奖励鼓励继续游戏
            reward = 1.0
        
        self.episode_reward += reward
        
        info = {
            "winner": winner,
            "episode_length": self.episode_length,
            "episode_reward": self.episode_reward,
            "last_move": (row, col)
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察状态
        
        Returns:
            编码后的棋盘状态
        """
        return self.board.get_encoded_state().astype(np.float32)
    
    def render(self, mode: str = 'human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
        """
        if mode == 'human':
            self.board.display()
            print(f"当前玩家: {'黑子' if self.board.current_player == 1 else '白子'}")
            print(f"回合数: {self.episode_length}")
            print("-" * 40)
    
    def get_valid_actions(self) -> np.ndarray:
        """
        获取有效动作掩码
        
        Returns:
            有效动作的布尔数组
        """
        valid_moves = self.board.get_valid_moves()
        valid_actions = np.zeros(self.board_size * self.board_size, dtype=bool)
        
        for row, col in valid_moves:
            action = row * self.board_size + col
            valid_actions[action] = True
            
        return valid_actions
    
    def get_action_mask(self) -> np.ndarray:
        """
        获取动作掩码（用于PPO）
        
        Returns:
            动作掩码，有效动作为1，无效动作为0
        """
        return self.get_valid_actions().astype(np.float32)
    
    def get_board_state(self) -> np.ndarray:
        """
        获取原始棋盘状态
        
        Returns:
            棋盘状态
        """
        return self.board.get_board_state()
    
    def copy(self):
        """复制环境"""
        new_env = GomokuEnvironment(self.board_size)
        new_env.board = self.board.copy()
        new_env.episode_reward = self.episode_reward
        new_env.episode_length = self.episode_length
        return new_env
