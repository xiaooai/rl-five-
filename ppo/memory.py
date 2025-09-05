"""
PPO经验回放缓冲区
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any


class PPOMemory:
    """PPO经验回放缓冲区"""
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 设备
        """
        self.capacity = capacity
        self.device = device
        
        # 存储经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
        
        # 计算优势函数和回报
        self.advantages = []
        self.returns = []
        
        self.size = 0
        self.ptr = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            value: float, log_prob: float, done: bool, action_mask: np.ndarray):
        """
        添加经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作对数概率
            done: 是否结束
            action_mask: 动作掩码
        """
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
            self.action_masks.append(action_mask)
            self.size += 1
        else:
            # 覆盖旧经验
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.dones[self.ptr] = done
            self.action_masks[self.ptr] = action_mask
            self.ptr = (self.ptr + 1) % self.capacity
    
    def compute_advantages_and_returns(self, gamma: float = 0.99, 
                                     lambda_gae: float = 0.95, 
                                     next_value: float = 0.0):
        """
        计算优势函数和回报
        
        Args:
            gamma: 折扣因子
            lambda_gae: GAE参数
            next_value: 下一状态的价值
        """
        # 转换为numpy数组
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])  # 添加下一状态价值
        dones = np.array(self.dones)
        
        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        获取批次数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次数据字典
        """
        # 随机采样索引
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # 转换为张量
        states = torch.FloatTensor([self.states[i] for i in indices]).to(self.device)
        actions = torch.LongTensor([self.actions[i] for i in indices]).to(self.device)
        old_log_probs = torch.FloatTensor([self.log_probs[i] for i in indices]).to(self.device)
        advantages = torch.FloatTensor([self.advantages[i] for i in indices]).to(self.device)
        returns = torch.FloatTensor([self.returns[i] for i in indices]).to(self.device)
        action_masks = torch.FloatTensor([self.action_masks[i] for i in indices]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'action_masks': action_masks
        }
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据
        
        Returns:
            所有数据字典
        """
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        action_masks = torch.FloatTensor(self.action_masks).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'action_masks': action_masks
        }
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.advantages.clear()
        self.returns.clear()
        
        self.size = 0
        self.ptr = 0
    
    def __len__(self):
        """返回缓冲区大小"""
        return self.size


class RolloutBuffer:
    """滚动缓冲区，用于存储一个episode的经验"""
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化滚动缓冲区
        
        Args:
            device: 设备
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
    
    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool, action_mask: np.ndarray):
        """
        添加经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作对数概率
            done: 是否结束
            action_mask: 动作掩码
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
    
    def compute_advantages_and_returns(self, gamma: float = 0.99,
                                     lambda_gae: float = 0.95,
                                     next_value: float = 0.0):
        """
        计算优势函数和回报
        
        Args:
            gamma: 折扣因子
            lambda_gae: GAE参数
            next_value: 下一状态的价值
        """
        # 转换为numpy数组
        rewards = np.array(self.rewards)
        # 确保values是标量列表
        values_list = [float(v) for v in self.values] + [float(next_value)]
        values = np.array(values_list)
        dones = np.array(self.dones)
        
        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据
        
        Returns:
            数据字典
        """
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        action_masks = torch.FloatTensor(self.action_masks).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'action_masks': action_masks
        }
    
    def __len__(self):
        """返回缓冲区大小"""
        return len(self.states)
