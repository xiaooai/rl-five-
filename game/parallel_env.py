"""
并行环境包装器
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from .environment import GomokuEnvironment


class ParallelEnvironment:
    """并行环境包装器，用于同时运行多个环境实例"""
    
    def __init__(self, num_envs: int, board_size: int = 15, device: str = 'cpu'):
        """
        初始化并行环境
        
        Args:
            num_envs: 并行环境数量
            board_size: 棋盘大小
            device: 设备
        """
        self.num_envs = num_envs
        self.board_size = board_size
        self.device = device
        
        # 创建多个环境实例
        self.envs = [GomokuEnvironment(board_size) for _ in range(num_envs)]
        
        # 环境状态
        self.states = np.zeros((num_envs, 3, board_size, board_size), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=int)
        
        # 重置所有环境
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置所有环境
        
        Returns:
            所有环境的初始状态 [num_envs, 3, board_size, board_size]
        """
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()
            self.dones[i] = False
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return self.states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        在所有环境中执行动作
        
        Args:
            actions: 动作数组 [num_envs]
            
        Returns:
            (观察, 奖励, 完成标志, 信息列表)
        """
        observations = np.zeros_like(self.states)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        
        for i, env in enumerate(self.envs):
            if not self.dones[i]:
                # 执行动作
                obs, reward, done, info = env.step(actions[i])
                
                observations[i] = obs
                rewards[i] = reward
                dones[i] = done
                infos.append(info)
                
                # 更新状态
                self.states[i] = obs
                self.dones[i] = done
                self.episode_rewards[i] += reward
                self.episode_lengths[i] += 1
                
                # 如果游戏结束，保存回合统计并重置环境
                if done:
                    # 更新info中的回合统计
                    info['episode_reward'] = self.episode_rewards[i]
                    info['episode_length'] = self.episode_lengths[i]
                    
                    self.states[i] = env.reset()
                    self.dones[i] = False
                    self.episode_rewards[i] = 0.0
                    self.episode_lengths[i] = 0
            else:
                # 环境已结束，保持当前状态
                observations[i] = self.states[i]
                rewards[i] = 0.0
                dones[i] = True
                infos.append({
                    "winner": 0,
                    "episode_length": 0,
                    "episode_reward": 0.0,
                    "last_move": None
                })
        
        return observations, rewards, dones, infos
    
    def get_action_masks(self) -> np.ndarray:
        """
        获取所有环境的动作掩码
        
        Returns:
            动作掩码数组 [num_envs, board_size * board_size]
        """
        masks = np.zeros((self.num_envs, self.board_size * self.board_size), dtype=np.float32)
        
        for i, env in enumerate(self.envs):
            if not self.dones[i]:
                masks[i] = env.get_action_mask()
            else:
                # 已结束的环境，所有动作都无效
                masks[i] = 0.0
        
        return masks
    
    def get_states(self) -> np.ndarray:
        """
        获取所有环境的当前状态
        
        Returns:
            状态数组 [num_envs, 3, board_size, board_size]
        """
        return self.states.copy()
    
    def get_episode_stats(self) -> Dict[str, np.ndarray]:
        """
        获取回合统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'dones': self.dones.copy()
        }
    
    def render(self, env_idx: int = 0, mode: str = 'human'):
        """
        渲染指定环境
        
        Args:
            env_idx: 环境索引
            mode: 渲染模式
        """
        if 0 <= env_idx < self.num_envs:
            self.envs[env_idx].render(mode)
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()


class VectorizedEnvironment:
    """向量化环境，使用批量操作提高效率"""
    
    def __init__(self, num_envs: int, board_size: int = 15, device: str = 'cpu'):
        """
        初始化向量化环境
        
        Args:
            num_envs: 并行环境数量
            board_size: 棋盘大小
            device: 设备
        """
        self.num_envs = num_envs
        self.board_size = board_size
        self.device = device
        
        # 创建多个环境实例
        self.envs = [GomokuEnvironment(board_size) for _ in range(num_envs)]
        
        # 环境状态
        self.states = np.zeros((num_envs, 3, board_size, board_size), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=int)
        
        # 重置所有环境
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置所有环境
        
        Returns:
            所有环境的初始状态 [num_envs, 3, board_size, board_size]
        """
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()
            self.dones[i] = False
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return self.states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        在所有环境中执行动作（向量化版本）
        
        Args:
            actions: 动作数组 [num_envs]
            
        Returns:
            (观察, 奖励, 完成标志, 信息列表)
        """
        observations = np.zeros_like(self.states)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        
        # 批量处理未结束的环境
        active_envs = ~self.dones
        active_indices = np.where(active_envs)[0]
        
        for i in active_indices:
            env = self.envs[i]
            obs, reward, done, info = env.step(actions[i])
            
            observations[i] = obs
            rewards[i] = reward
            dones[i] = done
            infos.append(info)
            
            # 更新状态
            self.states[i] = obs
            self.dones[i] = done
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            # 如果游戏结束，保存回合统计并重置环境
            if done:
                # 更新info中的回合统计
                info['episode_reward'] = self.episode_rewards[i]
                info['episode_length'] = self.episode_lengths[i]
                
                self.states[i] = env.reset()
                self.dones[i] = False
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
        
        # 处理已结束的环境
        for i in np.where(self.dones)[0]:
            observations[i] = self.states[i]
            rewards[i] = 0.0
            dones[i] = True
            infos.append({
                "winner": 0,
                "episode_length": 0,
                "episode_reward": 0.0,
                "last_move": None
            })
        
        return observations, rewards, dones, infos
    
    def get_action_masks(self) -> np.ndarray:
        """
        获取所有环境的动作掩码
        
        Returns:
            动作掩码数组 [num_envs, board_size * board_size]
        """
        masks = np.zeros((self.num_envs, self.board_size * self.board_size), dtype=np.float32)
        
        for i, env in enumerate(self.envs):
            if not self.dones[i]:
                masks[i] = env.get_action_mask()
            else:
                masks[i] = 0.0
        
        return masks
    
    def get_states(self) -> np.ndarray:
        """
        获取所有环境的当前状态
        
        Returns:
            状态数组 [num_envs, 3, board_size, board_size]
        """
        return self.states.copy()
    
    def get_episode_stats(self) -> Dict[str, np.ndarray]:
        """
        获取回合统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'dones': self.dones.copy()
        }
    
    def render(self, env_idx: int = 0, mode: str = 'human'):
        """
        渲染指定环境
        
        Args:
            env_idx: 环境索引
            mode: 渲染模式
        """
        if 0 <= env_idx < self.num_envs:
            self.envs[env_idx].render(mode)
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
