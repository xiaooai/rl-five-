"""
PPO神经网络模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, board_size: int = 15, hidden_dim: int = 512):
        """
        初始化策略网络
        
        Args:
            board_size: 棋盘大小
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.board_size = board_size
        self.action_dim = board_size * board_size
        
        # 卷积层提取特征
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * board_size * board_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层
        self.policy_head = nn.Linear(hidden_dim, self.action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态，形状为(batch_size, 3, board_size, board_size)
            
        Returns:
            (策略分布, 状态价值)
        """
        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, board_size: int = 15, hidden_dim: int = 512):
        """
        初始化Actor-Critic网络
        
        Args:
            board_size: 棋盘大小
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.policy_net = PolicyNetwork(board_size, hidden_dim)
        self.board_size = board_size
        self.action_dim = board_size * board_size
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            (动作概率, 动作对数概率, 状态价值)
        """
        policy_logits, value = self.policy_net(state)
        
        # 计算动作概率
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # 计算动作对数概率
        action_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        return action_probs, action_log_probs, value
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据策略选择动作
        
        Args:
            state: 输入状态
            action_mask: 动作掩码
            
        Returns:
            (选择的动作, 动作对数概率, 状态价值)
        """
        with torch.no_grad():
            policy_logits, value = self.policy_net(state)
            
            # 应用动作掩码
            if action_mask is not None:
                policy_logits = policy_logits + (action_mask - 1) * 1e8
            
            # 计算动作概率
            action_probs = F.softmax(policy_logits, dim=-1)
            
            # 采样动作
            action = torch.multinomial(action_probs, 1)
            
            # 计算动作对数概率
            action_log_probs = F.log_softmax(policy_logits, dim=-1)
            selected_log_probs = action_log_probs.gather(1, action)
            
            return action, selected_log_probs, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor, action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态和动作
        
        Args:
            state: 输入状态
            action: 动作
            action_mask: 动作掩码
            
        Returns:
            (动作对数概率, 状态价值, 策略熵)
        """
        policy_logits, value = self.policy_net(state)
        
        # 应用动作掩码
        if action_mask is not None:
            policy_logits = policy_logits + (action_mask - 1) * 1e8
        
        # 计算动作对数概率
        action_log_probs = F.log_softmax(policy_logits, dim=-1)
        # 确保 action 有正确的维度用于 gather 操作
        if action.dim() == 1:
            action = action.unsqueeze(1)
        selected_log_probs = action_log_probs.gather(1, action)
        
        # 计算策略熵
        action_probs = F.softmax(policy_logits, dim=-1)
        entropy = -(action_probs * action_log_probs).sum(dim=-1, keepdim=True)
        
        return selected_log_probs, value, entropy


class ValueNetwork(nn.Module):
    """价值网络（可选，用于更稳定的价值估计）"""
    
    def __init__(self, board_size: int = 15, hidden_dim: int = 512):
        """
        初始化价值网络
        
        Args:
            board_size: 棋盘大小
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.board_size = board_size
        
        # 卷积层
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * board_size * board_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            状态价值
        """
        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出价值
        value = self.value_head(x)
        
        return value
