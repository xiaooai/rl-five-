"""
五子棋棋盘逻辑 - 规范化版本
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class Player(Enum):
    """玩家枚举"""
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class GameState(Enum):
    """游戏状态枚举"""
    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = -1
    DRAW = 2
    INVALID_MOVE = 3


class Board:
    """五子棋棋盘类 - 规范化版本"""
    
    def __init__(self, size: int = 15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = Player.BLACK
        self.move_count = 0
        self.move_history = []  # 记录所有移动历史
        self.last_move = None
        self.game_state = GameState.ONGOING
        
        # 游戏统计
        self.black_stones = 0
        self.white_stones = 0
        
    def reset(self):
        """重置棋盘"""
        self.board.fill(Player.EMPTY.value)
        self.current_player = Player.BLACK
        self.move_count = 0
        self.move_history.clear()
        self.last_move = None
        self.game_state = GameState.ONGOING
        self.black_stones = 0
        self.white_stones = 0
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        检查落子是否有效
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            是否有效
        """
        # 检查坐标范围
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
            
        # 检查位置是否为空
        if self.board[row, col] != Player.EMPTY.value:
            return False
            
        # 检查游戏是否已结束
        if self.game_state != GameState.ONGOING:
            return False
            
        return True
    
    def make_move(self, row: int, col: int) -> Tuple[bool, GameState]:
        """
        落子
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            (是否成功落子, 游戏状态)
        """
        if not self.is_valid_move(row, col):
            self.game_state = GameState.INVALID_MOVE
            return False, self.game_state
            
        # 执行落子
        self.board[row, col] = self.current_player.value
        self.last_move = (row, col)
        self.move_history.append((row, col, self.current_player.value))
        self.move_count += 1
        
        # 更新统计
        if self.current_player == Player.BLACK:
            self.black_stones += 1
        else:
            self.white_stones += 1
        
        # 检查游戏状态
        self.game_state = self._check_game_state(row, col)
        
        # 切换玩家
        if self.game_state == GameState.ONGOING:
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        
        return True, self.game_state
    
    def _check_game_state(self, row: int, col: int) -> GameState:
        """
        检查游戏状态
        
        Args:
            row: 最后落子的行
            col: 最后落子的列
            
        Returns:
            游戏状态
        """
        player = self.board[row, col]
        
        # 检查四个方向：水平、垂直、对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # 包含当前棋子
            
            # 向一个方向计数
            r, c = row + dr, col + dc
            while (0 <= r < self.size and 0 <= c < self.size and 
                   self.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # 向相反方向计数
            r, c = row - dr, col - dc
            while (0 <= r < self.size and 0 <= c < self.size and 
                   self.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 5:
                return GameState.BLACK_WIN if player == Player.BLACK.value else GameState.WHITE_WIN
        
        # 检查是否平局
        if self.move_count >= self.size * self.size:
            return GameState.DRAW
            
        return GameState.ONGOING
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        获取所有有效落子位置
        
        Returns:
            有效位置列表
        """
        valid_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves
    
    def get_valid_moves_near_stones(self, radius: int = 2) -> List[Tuple[int, int]]:
        """
        获取已落子附近的有效位置（用于优化搜索）
        
        Args:
            radius: 搜索半径
            
        Returns:
            有效位置列表
        """
        if self.move_count == 0:
            # 如果棋盘为空，返回中心位置
            center = self.size // 2
            return [(center, center)]
        
        valid_moves = set()
        
        # 遍历所有已落子的位置
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] != Player.EMPTY.value:
                    # 在已落子位置周围搜索
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            new_row, new_col = row + dr, col + dc
                            if self.is_valid_move(new_row, new_col):
                                valid_moves.add((new_row, new_col))
        
        return list(valid_moves)
    
    def get_board_state(self) -> np.ndarray:
        """
        获取当前棋盘状态
        
        Returns:
            棋盘状态数组
        """
        return self.board.copy()
    
    def get_encoded_state(self) -> np.ndarray:
        """
        获取编码后的棋盘状态（用于神经网络输入）
        
        Returns:
            编码后的状态，形状为(4, size, size)
        """
        # 创建四个通道：黑子、白子、当前玩家、空位
        black_channel = (self.board == Player.BLACK.value).astype(float)
        white_channel = (self.board == Player.WHITE.value).astype(float)
        current_player_channel = np.full((self.size, self.size), 
                                       self.current_player.value, dtype=float)
        empty_channel = (self.board == Player.EMPTY.value).astype(float)
        
        return np.stack([black_channel, white_channel, current_player_channel, empty_channel])
    
    def get_advanced_state(self) -> np.ndarray:
        """
        获取高级状态表示（包含更多信息）
        
        Returns:
            高级状态，形状为(6, size, size)
        """
        # 基础状态
        black_channel = (self.board == Player.BLACK.value).astype(float)
        white_channel = (self.board == Player.WHITE.value).astype(float)
        current_player_channel = np.full((self.size, self.size), 
                                       self.current_player.value, dtype=float)
        empty_channel = (self.board == Player.EMPTY.value).astype(float)
        
        # 添加位置信息
        position_channel = np.zeros((self.size, self.size), dtype=float)
        for i in range(self.size):
            for j in range(self.size):
                # 距离中心的距离
                center = self.size // 2
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                position_channel[i, j] = distance / (self.size // 2)
        
        # 添加移动历史信息
        history_channel = np.zeros((self.size, self.size), dtype=float)
        for i, (row, col, player) in enumerate(self.move_history):
            history_channel[row, col] = (i + 1) / len(self.move_history) if self.move_history else 0
        
        return np.stack([black_channel, white_channel, current_player_channel, 
                        empty_channel, position_channel, history_channel])
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        获取游戏信息
        
        Returns:
            游戏信息字典
        """
        return {
            'current_player': self.current_player.value,
            'move_count': self.move_count,
            'game_state': self.game_state.value,
            'last_move': self.last_move,
            'black_stones': self.black_stones,
            'white_stones': self.white_stones,
            'valid_moves_count': len(self.get_valid_moves()),
            'board_size': self.size
        }
    
    def display(self):
        """显示棋盘"""
        print("   ", end="")
        for col in range(self.size):
            print(f"{col:2d}", end="")
        print()
        
        for row in range(self.size):
            print(f"{row:2d} ", end="")
            for col in range(self.size):
                if self.board[row, col] == Player.BLACK.value:
                    print("●", end=" ")
                elif self.board[row, col] == Player.WHITE.value:
                    print("○", end=" ")
                else:
                    print("·", end=" ")
            print()
        
        print(f"当前玩家: {'黑子' if self.current_player == Player.BLACK else '白子'}")
        print(f"移动次数: {self.move_count}")
        print(f"游戏状态: {self.game_state.name}")
    
    def copy(self):
        """复制棋盘"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.move_history = self.move_history.copy()
        new_board.last_move = self.last_move
        new_board.game_state = self.game_state
        new_board.black_stones = self.black_stones
        new_board.white_stones = self.white_stones
        return new_board
    
    def __str__(self):
        """字符串表示"""
        return f"Board(size={self.size}, moves={self.move_count}, state={self.game_state.name})"
    
    def __repr__(self):
        """详细字符串表示"""
        return (f"Board(size={self.size}, current_player={self.current_player.value}, "
                f"moves={self.move_count}, state={self.game_state.name}, "
                f"last_move={self.last_move})")