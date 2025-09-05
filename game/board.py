"""
五子棋棋盘逻辑
"""
import numpy as np
from typing import List, Tuple, Optional


class Board:
    """五子棋棋盘类"""
    
    def __init__(self, size: int = 15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1表示黑子，-1表示白子
        self.move_count = 0
        self.last_move = None
        
    def reset(self):
        """重置棋盘"""
        self.board.fill(0)
        self.current_player = 1
        self.move_count = 0
        self.last_move = None
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        检查落子是否有效
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            是否有效
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        return self.board[row, col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """
        落子
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            是否成功落子
        """
        if not self.is_valid_move(row, col):
            return False
            
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        self.current_player *= -1
        return True
    
    def check_winner(self) -> int:
        """
        检查是否有获胜者
        
        Returns:
            1: 黑子获胜, -1: 白子获胜, 0: 继续游戏, 2: 平局
        """
        if self.last_move is None:
            return 0
            
        row, col = self.last_move
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
                return player
        
        # 检查是否平局
        if self.move_count >= self.size * self.size:
            return 2
            
        return 0
    
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
            编码后的状态，形状为(3, size, size)
        """
        # 创建三个通道：黑子、白子、当前玩家
        black_channel = (self.board == 1).astype(float)
        white_channel = (self.board == -1).astype(float)
        current_player_channel = np.full((self.size, self.size), 
                                       self.current_player, dtype=float)
        
        return np.stack([black_channel, white_channel, current_player_channel])
    
    def display(self):
        """显示棋盘"""
        print("   ", end="")
        for col in range(self.size):
            print(f"{col:2d}", end="")
        print()
        
        for row in range(self.size):
            print(f"{row:2d} ", end="")
            for col in range(self.size):
                if self.board[row, col] == 1:
                    print("●", end=" ")
                elif self.board[row, col] == -1:
                    print("○", end=" ")
                else:
                    print("·", end=" ")
            print()
    
    def copy(self):
        """复制棋盘"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.last_move = self.last_move
        return new_board
