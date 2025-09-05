"""
五子棋游戏环境 - 规范化版本
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from .board import Board, Player, GameState


class GomokuEnvironment(gym.Env):
    """五子棋游戏环境 - 规范化版本"""
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'ansi'],
        'render_fps': 4,
    }
    
    def __init__(self, 
                 board_size: int = 15,
                 render_mode: Optional[str] = None,
                 reward_type: str = 'standard',
                 invalid_move_penalty: float = -10.0,
                 win_reward: float = 100.0,
                 lose_reward: float = -100.0,
                 draw_reward: float = 0.0,
                 step_reward: float = 0.1,
                 use_advanced_state: bool = False,
                 max_moves: Optional[int] = None,
                 allow_invalid_moves: bool = False):
        """
        初始化环境
        
        Args:
            board_size: 棋盘大小
            render_mode: 渲染模式
            reward_type: 奖励类型 ('standard', 'sparse', 'dense')
            invalid_move_penalty: 无效移动惩罚
            win_reward: 获胜奖励
            lose_reward: 失败奖励
            draw_reward: 平局奖励
            step_reward: 每步奖励
            use_advanced_state: 是否使用高级状态表示
            max_moves: 最大移动次数
            allow_invalid_moves: 是否允许无效移动
        """
        super().__init__()
        
        self.board_size = board_size
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.invalid_move_penalty = invalid_move_penalty
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.step_reward = step_reward
        self.use_advanced_state = use_advanced_state
        self.max_moves = max_moves
        self.allow_invalid_moves = allow_invalid_moves
        
        # 创建棋盘
        self.board = Board(board_size)
        
        # 动作空间：所有可能的落子位置
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 观察空间：编码后的棋盘状态
        if use_advanced_state:
            obs_shape = (6, board_size, board_size)
        else:
            obs_shape = (4, board_size, board_size)
            
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
        
        # 游戏统计
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_info = {}
        
        # 渲染相关
        self.window = None
        self.clock = None
        
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            (初始观察, 信息)
        """
        super().reset(seed=seed)
        
        # 重置棋盘
        self.board.reset()
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_info = {}
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作索引
            
        Returns:
            (观察, 奖励, 终止, 截断, 信息)
        """
        # 将动作索引转换为坐标
        row = action // self.board_size
        col = action % self.board_size
        
        # 执行移动
        success, game_state = self.board.make_move(row, col)
        self.episode_length += 1
        
        # 计算奖励
        reward = self._calculate_reward(success, game_state, row, col)
        self.episode_reward += reward
        
        # 判断是否结束
        terminated = game_state in [GameState.BLACK_WIN, GameState.WHITE_WIN, GameState.DRAW]
        truncated = False  # 可以添加最大步数限制
        
        # 获取观察和信息
        observation = self._get_observation()
        info = self._get_info()
        
        # 添加额外信息
        info.update({
            'success': success,
            'game_state': game_state.value,
            'last_move': (row, col),
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'current_player': self.board.current_player.value,
            'valid_moves_count': len(self.board.get_valid_moves())
        })
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, success: bool, game_state: GameState, row: int, col: int) -> float:
        """
        计算奖励
        
        Args:
            success: 移动是否成功
            game_state: 游戏状态
            row: 行坐标
            col: 列坐标
            
        Returns:
            奖励值
        """
        if not success:
            return self.invalid_move_penalty
        
        if self.reward_type == 'sparse':
            # 稀疏奖励：只在游戏结束时给予奖励
            if game_state == GameState.BLACK_WIN:
                return self.win_reward
            elif game_state == GameState.WHITE_WIN:
                return self.lose_reward
            elif game_state == GameState.DRAW:
                return self.draw_reward
            else:
                return 0.0
                
        elif self.reward_type == 'dense':
            # 密集奖励：考虑位置价值、威胁等
            reward = 0.0
            
            # 基础步奖励
            reward += self.step_reward
            
            # 位置价值奖励
            center = self.board_size // 2
            distance_from_center = np.sqrt((row - center) ** 2 + (col - center) ** 2)
            position_reward = max(0, 1.0 - distance_from_center / (self.board_size // 2))
            reward += position_reward * 0.1
            
            # 威胁检测奖励
            threat_reward = self._calculate_threat_reward(row, col)
            reward += threat_reward
            
            # 游戏结束奖励
            if game_state == GameState.BLACK_WIN:
                reward += self.win_reward
            elif game_state == GameState.WHITE_WIN:
                reward += self.lose_reward
            elif game_state == GameState.DRAW:
                reward += self.draw_reward
                
            return reward
            
        else:  # standard
            # 标准奖励
            if game_state == GameState.BLACK_WIN:
                return self.win_reward
            elif game_state == GameState.WHITE_WIN:
                return self.lose_reward
            elif game_state == GameState.DRAW:
                return self.draw_reward
            else:
                return self.step_reward
    
    def _calculate_threat_reward(self, row: int, col: int) -> float:
        """
        计算威胁奖励（检测连子情况）
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            威胁奖励
        """
        player = self.board.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_reward = 0.0
        
        for dr, dc in directions:
            count = 1
            
            # 向一个方向计数
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.board.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # 向相反方向计数
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.board.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            # 根据连子数量给予奖励
            if count >= 5:
                total_reward += 10.0  # 五连
            elif count == 4:
                total_reward += 5.0   # 四连
            elif count == 3:
                total_reward += 2.0   # 三连
            elif count == 2:
                total_reward += 0.5   # 二连
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察状态
        
        Returns:
            编码后的棋盘状态
        """
        if self.use_advanced_state:
            return self.board.get_advanced_state().astype(np.float32)
        else:
            return self.board.get_encoded_state().astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        获取环境信息
        
        Returns:
            信息字典
        """
        return {
            'game_info': self.board.get_game_info(),
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'valid_moves': self.get_valid_actions().tolist()
        }
    
    def render(self, mode: str = 'human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
        """
        if mode == 'human':
            self.board.display()
            print(f"回合数: {self.episode_length}")
            print(f"累计奖励: {self.episode_reward:.2f}")
            print("-" * 40)
            
        elif mode == 'ansi':
            return self._render_ansi()
            
        elif mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_ansi(self) -> str:
        """ANSI渲染"""
        output = []
        output.append("   " + "".join(f"{col:2d}" for col in range(self.board_size)))
        
        for row in range(self.board_size):
            line = f"{row:2d} "
            for col in range(self.board_size):
                if self.board.board[row, col] == Player.BLACK.value:
                    line += "● "
                elif self.board.board[row, col] == Player.WHITE.value:
                    line += "○ "
                else:
                    line += "· "
            output.append(line)
        
        return "\n".join(output)
    
    def _render_rgb_array(self) -> np.ndarray:
        """RGB数组渲染"""
        # 简化的RGB渲染，实际应用中可以使用更复杂的渲染
        rgb_array = np.zeros((self.board_size * 20, self.board_size * 20, 3), dtype=np.uint8)
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                y_start, y_end = row * 20, (row + 1) * 20
                x_start, x_end = col * 20, (col + 1) * 20
                
                if self.board.board[row, col] == Player.BLACK.value:
                    rgb_array[y_start:y_end, x_start:x_end] = [0, 0, 0]  # 黑色
                elif self.board.board[row, col] == Player.WHITE.value:
                    rgb_array[y_start:y_end, x_start:x_end] = [255, 255, 255]  # 白色
                else:
                    rgb_array[y_start:y_end, x_start:x_end] = [200, 200, 200]  # 灰色
        
        return rgb_array
    
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
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        获取游戏信息
        
        Returns:
            游戏信息
        """
        return self.board.get_game_info()
    
    def copy(self):
        """复制环境"""
        new_env = GomokuEnvironment(
            board_size=self.board_size,
            render_mode=self.render_mode,
            reward_type=self.reward_type,
            invalid_move_penalty=self.invalid_move_penalty,
            win_reward=self.win_reward,
            lose_reward=self.lose_reward,
            draw_reward=self.draw_reward,
            step_reward=self.step_reward,
            use_advanced_state=self.use_advanced_state
        )
        new_env.board = self.board.copy()
        new_env.episode_reward = self.episode_reward
        new_env.episode_length = self.episode_length
        new_env.episode_info = self.episode_info.copy()
        return new_env
    
    def close(self):
        """关闭环境"""
        if self.window is not None:
            self.window.close()
        if self.clock is not None:
            self.clock = None
    
    def __str__(self):
        """字符串表示"""
        return f"GomokuEnvironment(board_size={self.board_size}, reward_type={self.reward_type})"
    
    def __repr__(self):
        """详细字符串表示"""
        return (f"GomokuEnvironment(board_size={self.board_size}, "
                f"reward_type={self.reward_type}, episode_length={self.episode_length}, "
                f"episode_reward={self.episode_reward:.2f})")