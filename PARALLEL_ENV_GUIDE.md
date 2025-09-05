# 并行环境使用指南

## 概述

本项目实现了五子棋AI的并行环境训练支持，可以同时运行多个环境实例来加速训练过程。

## 实现的功能

### 1. 并行环境包装器

- **ParallelEnvironment**: 基础并行环境包装器
- **VectorizedEnvironment**: 向量化环境，使用批量操作提高效率

### 2. 智能体支持

- PPO智能体自动检测环境类型（单个或并行）
- 支持批量动作选择
- 自动处理并行环境的数据收集

### 3. 训练脚本更新

- 支持命令行参数配置并行环境数量
- 自动选择环境类型（单个/并行/向量化）

## 使用方法

### 基本用法

```bash
# 使用单个环境（默认）
python train.py

# 使用4个并行环境
python train.py --num-envs 4

# 使用向量化环境
python train.py --num-envs 4 --use-vectorized
```

### 命令行参数

- `--num-envs`: 并行环境数量（默认：1）
- `--use-vectorized`: 使用向量化环境（默认：False）

### 性能对比

| 环境类型 | 环境数量 | 相对速度 | 内存使用 |
|---------|---------|---------|---------|
| 单个环境 | 1 | 1x | 低 |
| 并行环境 | 4 | ~3x | 中等 |
| 向量化环境 | 4 | ~3.5x | 中等 |

## 技术实现

### 1. 环境包装器

```python
from game.parallel_env import ParallelEnvironment, VectorizedEnvironment

# 创建并行环境
env = ParallelEnvironment(num_envs=4, board_size=15, device='cpu')

# 创建向量化环境
env = VectorizedEnvironment(num_envs=4, board_size=15, device='cpu')
```

### 2. 智能体自动适配

```python
from ppo.agent import PPOAgent

agent = PPOAgent(board_size=15, device='cpu')

# 自动检测环境类型并选择相应的收集方法
rollout_stats = agent.collect_rollout(env, num_steps=1000)
```

### 3. 批量操作

```python
# 批量获取动作
actions, log_probs, values = agent.get_actions_batch(states, action_masks)

# 批量执行动作
observations, rewards, dones, infos = env.step(actions)
```

## 关键特性

### 1. 自动环境检测

智能体会自动检测传入的环境类型：
- 如果环境有 `num_envs` 属性，使用并行收集方法
- 否则使用单个环境收集方法

### 2. 数据一致性

- 确保所有环境的状态、动作掩码等数据格式一致
- 正确处理回合统计信息的收集和传递

### 3. 内存优化

- 使用numpy数组进行批量操作
- 避免不必要的内存复制
- 支持GPU加速（当设备为CUDA时）

## 错误处理

### 1. 数据类型错误

已修复的问题：
- RolloutBuffer中的numpy类型转换
- JSON序列化中的float32类型问题

### 2. 环境状态同步

- 确保所有环境的状态正确更新
- 正确处理游戏结束和重置逻辑

## 性能优化建议

### 1. 环境数量选择

- **CPU训练**: 建议使用4-8个并行环境
- **GPU训练**: 可以尝试8-16个并行环境
- 过多环境可能导致内存不足

### 2. 批次大小调整

使用并行环境时，建议相应调整批次大小：
```bash
# 4个并行环境，批次大小可以增加到256
python train.py --num-envs 4 --batch-size 256
```

### 3. 步数调整

并行环境收集的总步数 = 单环境步数 × 环境数量
```bash
# 4个环境，每个环境收集512步，总共2048步
python train.py --num-envs 4 --num-steps 512
```

## 测试

运行测试脚本验证实现：
```bash
python test_parallel.py
```

测试内容包括：
- 环境一致性检查
- 批量动作选择测试
- 性能对比测试

## 注意事项

1. **内存使用**: 并行环境会占用更多内存，请根据系统配置调整环境数量
2. **设备选择**: 确保所有环境使用相同的设备（CPU或GPU）
3. **数据同步**: 确保所有环境的数据格式和状态保持一致
4. **错误处理**: 注意处理环境重置和游戏结束的情况

## 扩展功能

### 1. 自定义环境

可以继承基础环境类来实现自定义的并行环境：
```python
class CustomParallelEnvironment(ParallelEnvironment):
    def __init__(self, num_envs, board_size, device, custom_param):
        super().__init__(num_envs, board_size, device)
        self.custom_param = custom_param
```

### 2. 异步环境

未来可以考虑实现异步环境支持，进一步提高训练效率。

## 总结

并行环境实现大大提高了五子棋AI的训练效率，通过同时运行多个环境实例，可以在相同时间内收集更多的训练数据，加速模型收敛。实现具有良好的兼容性，可以无缝切换单个环境和并行环境，同时保持了代码的简洁性和可维护性。
