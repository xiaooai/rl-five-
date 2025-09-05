"""
PPO智能体
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any
import math

from model.network import ActorCritic
from ppo.memory import RolloutBuffer


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, 
                 board_size: int = 15,
                 hidden_dim: int = 512,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        初始化PPO智能体
        
        Args:
            board_size: 棋盘大小
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            lambda_gae: GAE参数
            clip_ratio: PPO裁剪比例
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
        """
        self.board_size = board_size
        self.device = device
        
        # 创建网络
        self.actor_critic = ActorCritic(board_size, hidden_dim).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO参数
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 经验缓冲区
        self.rollout_buffer = RolloutBuffer(device)
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray = None) -> Tuple[int, float, float]:
        """
        根据策略选择动作
        
        Args:
            state: 当前状态
            action_mask: 动作掩码
            
        Returns:
            (动作, 动作对数概率, 状态价值)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            action_mask_tensor = None
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(
                state_tensor, action_mask_tensor
            )
        
        return action.item(), log_prob.item(), value.item()
    
    def collect_rollout(self, env, num_steps: int) -> Dict[str, Any]:
        """
        收集经验数据
        
        Args:
            env: 环境（单个环境或并行环境）
            num_steps: 收集步数
            
        Returns:
            收集统计信息
        """
        # 检查是否为并行环境
        if hasattr(env, 'num_envs'):
            return self._collect_rollout_parallel(env, num_steps)
        else:
            return self._collect_rollout_single(env, num_steps)
    
    def _collect_rollout_single(self, env, num_steps: int) -> Dict[str, Any]:
        """
        单个环境收集经验数据
        
        Args:
            env: 单个环境
            num_steps: 收集步数
            
        Returns:
            收集统计信息
        """
        self.rollout_buffer.reset()
        
        state = env.reset()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(num_steps):
            # 获取动作掩码
            action_mask = env.get_action_mask()
            
            # 选择动作
            action, log_prob, value = self.get_action(state, action_mask)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            self.rollout_buffer.add(
                state, action, reward, value, log_prob, done, action_mask
            )
            
            current_episode_reward += reward
            current_episode_length += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                state = env.reset()
            else:
                state = next_state
        
        # 计算最后状态的价值
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, last_value = self.actor_critic.get_action(last_state_tensor)
            last_value = last_value.item()
        
        # 计算优势函数和回报
        self.rollout_buffer.compute_advantages_and_returns(
            self.gamma, self.lambda_gae, last_value
        )
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': num_steps
        }
    
    def _collect_rollout_parallel(self, env, num_steps: int) -> Dict[str, Any]:
        """
        并行环境收集经验数据
        
        Args:
            env: 并行环境
            num_steps: 收集步数
            
        Returns:
            收集统计信息
        """
        self.rollout_buffer.reset()
        
        # 重置所有环境
        states = env.reset()
        episode_rewards = []
        episode_lengths = []
        
        for step in range(num_steps):
            # 获取所有环境的动作掩码
            action_masks = env.get_action_masks()
            
            # 批量选择动作
            actions, log_probs, values = self.get_actions_batch(states, action_masks)
            
            # 在所有环境中执行动作
            next_states, rewards, dones, infos = env.step(actions)
            
            # 批量存储经验
            for i in range(env.num_envs):
                self.rollout_buffer.add(
                    states[i], actions[i], rewards[i], values[i], 
                    log_probs[i], dones[i], action_masks[i]
                )
            
            # 收集完成的回合统计
            for i, info in enumerate(infos):
                if dones[i] and 'episode_reward' in info:
                    episode_rewards.append(info['episode_reward'])
                    episode_lengths.append(info['episode_length'])
            
            states = next_states
        
        # 计算最后状态的价值
        with torch.no_grad():
            last_states_tensor = torch.FloatTensor(states).to(self.device)
            _, _, last_values = self.actor_critic.get_action(last_states_tensor)
            last_value = last_values.mean().item()
        
        # 计算优势函数和回报
        self.rollout_buffer.compute_advantages_and_returns(
            self.gamma, self.lambda_gae, last_value
        )
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': num_steps * env.num_envs
        }
    
    def get_actions_batch(self, states: np.ndarray, action_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量获取动作
        
        Args:
            states: 状态数组 [batch_size, 3, board_size, board_size]
            action_masks: 动作掩码数组 [batch_size, action_space_size]
            
        Returns:
            (动作数组, 对数概率数组, 价值数组)
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        action_masks_tensor = torch.FloatTensor(action_masks).to(self.device)
        
        with torch.no_grad():
            actions, log_probs, values = self.actor_critic.get_action(
                states_tensor, action_masks_tensor
            )
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            num_epochs: 更新轮数
            batch_size: 批次大小
            
        Returns:
            训练统计信息
        """
        if len(self.rollout_buffer) == 0:
            return {}
        
        # 获取所有数据
        data = self.rollout_buffer.get_data()
        
        # 计算批次数量
        num_batches = math.ceil(len(self.rollout_buffer) / batch_size)
        
        # 存储统计信息
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(self.rollout_buffer))
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.rollout_buffer))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取批次数据
                batch_data = {
                    key: value[batch_indices] for key, value in data.items()
                }
                
                # 计算损失
                loss_info = self._compute_loss(batch_data)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss_info['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                # 记录统计信息
                for key, value in loss_info.items():
                    if key != 'total_loss':
                        epoch_stats[key].append(value.item())
                    else:
                        epoch_stats[key].append(value.item())
        
        # 计算平均统计信息
        avg_stats = {}
        for key, values in epoch_stats.items():
            if values:
                avg_stats[key] = np.mean(values)
                self.training_stats[key].append(avg_stats[key])
        
        return avg_stats
    
    def _compute_loss(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失
        
        Args:
            batch_data: 批次数据
            
        Returns:
            损失字典
        """
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['old_log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']
        action_masks = batch_data['action_masks']
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 前向传播
        new_log_probs, values, entropy = self.actor_critic.evaluate_actions(
            states, actions, action_masks
        )
        
        # 计算概率比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 计算熵损失
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # 计算KL散度
        kl_divergence = (old_log_probs - new_log_probs).mean()
        
        # 计算裁剪比例
        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'kl_divergence': kl_divergence,
            'clip_fraction': clip_fraction
        }
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device,weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def get_training_stats(self) -> Dict[str, list]:
        """
        获取训练统计信息
        
        Returns:
            训练统计信息
        """
        return self.training_stats.copy()
