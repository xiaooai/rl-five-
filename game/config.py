"""
五子棋环境配置
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class GomokuConfig:
    """五子棋环境配置类"""
    
    # 基础参数
    board_size: int = 15
    render_mode: Optional[str] = None
    
    # 奖励参数
    reward_type: str = 'standard'  # 'standard', 'sparse', 'dense'
    invalid_move_penalty: float = -10.0
    win_reward: float = 100.0
    lose_reward: float = -100.0
    draw_reward: float = 0.0
    step_reward: float = 0.1
    
    # 状态表示
    use_advanced_state: bool = False
    
    # 游戏规则
    max_moves: Optional[int] = None  # 最大移动次数，None表示无限制
    allow_invalid_moves: bool = False  # 是否允许无效移动
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GomokuConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'board_size': self.board_size,
            'render_mode': self.render_mode,
            'reward_type': self.reward_type,
            'invalid_move_penalty': self.invalid_move_penalty,
            'win_reward': self.win_reward,
            'lose_reward': self.lose_reward,
            'draw_reward': self.draw_reward,
            'step_reward': self.step_reward,
            'use_advanced_state': self.use_advanced_state,
            'max_moves': self.max_moves,
            'allow_invalid_moves': self.allow_invalid_moves
        }


# 预定义配置
STANDARD_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='standard',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=1.0
)

SPARSE_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='sparse',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=0.0
)

DENSE_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='dense',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=0.1,
    use_advanced_state=True
)

SMALL_BOARD_CONFIG = GomokuConfig(
    board_size=9,
    reward_type='standard',
    win_reward=50.0,
    lose_reward=-50.0,
    step_reward=0.5
)

LARGE_BOARD_CONFIG = GomokuConfig(
    board_size=19,
    reward_type='dense',
    win_reward=200.0,
    lose_reward=-200.0,
    step_reward=0.05,
    use_advanced_state=True
)
