# 规范化五子棋环境使用指南

## 概述

我们已经将五子棋环境进行了全面规范化，使其更符合标准的强化学习环境设计。新环境具有更好的模块化、更丰富的功能和更标准化的接口。

## 主要改进

### 1. **更完善的游戏规则**
- 使用枚举类型定义玩家和游戏状态
- 更严格的游戏逻辑验证
- 完整的移动历史记录
- 游戏统计信息

### 2. **更丰富的状态表示**
- **标准状态**: 4通道 (黑子、白子、当前玩家、空位)
- **高级状态**: 6通道 (增加位置信息和历史信息)
- 支持不同的状态表示模式

### 3. **更合理的奖励设计**
- **标准奖励**: 基础的游戏奖励
- **稀疏奖励**: 只在游戏结束时给予奖励
- **密集奖励**: 包含位置价值和威胁检测

### 4. **更标准的环境接口**
- 符合Gymnasium标准
- 支持多种渲染模式
- 完整的元数据信息
- 标准化的reset和step接口

## 使用方法

### 基本使用

```python
from game.factory import create_gomoku_env

# 创建标准环境
env = create_gomoku_env('standard')

# 重置环境
obs, info = env.reset()

# 执行动作
obs, reward, terminated, truncated, info = env.step(action)

# 关闭环境
env.close()
```

### 不同配置的环境

```python
# 标准配置
env = create_gomoku_env('standard')

# 稀疏奖励配置
env = create_gomoku_env('sparse')

# 密集奖励配置
env = create_gomoku_env('dense')

# 小棋盘配置
env = create_gomoku_env('small')

# 大棋盘配置
env = create_gomoku_env('large')
```

### 自定义配置

```python
from game.config import GomokuConfig

# 创建自定义配置
config = GomokuConfig(
    board_size=9,
    reward_type='dense',
    use_advanced_state=True,
    win_reward=50.0,
    step_reward=0.1
)

# 使用配置创建环境
env = GomokuEnvironment(**config.to_dict())
```

## 训练脚本更新

### 新的命令行参数

```bash
python train.py 
    --env-config standard 
    --reward-type dense 
    --use-advanced-state 
    --board-size 15 
    --num-envs 4
```

### 参数说明

- `--env-config`: 环境配置类型 (standard, sparse, dense, small, large)
- `--reward-type`: 奖励类型 (standard, sparse, dense)
- `--use-advanced-state`: 使用高级状态表示
- `--board-size`: 棋盘大小

## 环境配置详解

### 1. **标准配置 (standard)**
```python
STANDARD_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='standard',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=1.0
)
```

### 2. **稀疏奖励配置 (sparse)**
```python
SPARSE_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='sparse',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=0.0
)
```

### 3. **密集奖励配置 (dense)**
```python
DENSE_CONFIG = GomokuConfig(
    board_size=15,
    reward_type='dense',
    win_reward=100.0,
    lose_reward=-100.0,
    step_reward=0.1,
    use_advanced_state=True
)
```

## 状态表示

### 标准状态 (4通道)
- 通道0: 黑子位置
- 通道1: 白子位置  
- 通道2: 当前玩家
- 通道3: 空位

### 高级状态 (6通道)
- 通道0-3: 同标准状态
- 通道4: 位置信息 (距离中心的距离)
- 通道5: 历史信息 (移动顺序)

## 奖励系统

### 标准奖励
- 获胜: +100
- 失败: -100
- 平局: 0
- 每步: +1

### 稀疏奖励
- 只在游戏结束时给予奖励
- 适合需要明确胜负信号的任务

### 密集奖励
- 包含位置价值奖励
- 包含威胁检测奖励
- 更细粒度的学习信号

## 并行环境支持

### 创建并行环境

```python
from game.parallel_env import ParallelEnvironment, VectorizedEnvironment

# 并行环境
env = ParallelEnvironment(
    num_envs=4,
    board_size=15,
    env_config='standard',
    reward_type='dense'
)

# 向量化环境
env = VectorizedEnvironment(
    num_envs=4,
    board_size=15,
    env_config='standard',
    reward_type='dense'
)
```

## 测试和验证

### 运行测试

```bash
python test_improved_env.py
```

### 测试内容
- 基本功能测试
- 不同配置测试
- 高级状态测试
- 奖励类型测试
- 游戏逻辑测试
- 渲染功能测试
- 工厂类测试

## 迁移指南

### 从旧环境迁移

1. **更新导入**:
```python
# 旧版本
from game.environment import GomokuEnvironment

# 新版本
from game.factory import create_gomoku_env
```

2. **更新环境创建**:
```python
# 旧版本
env = GomokuEnvironment(board_size=15)

# 新版本
env = create_gomoku_env('standard', board_size=15)
```

3. **更新reset调用**:
```python
# 旧版本
obs = env.reset()

# 新版本
obs, info = env.reset()
```

4. **更新step调用**:
```python
# 旧版本
obs, reward, done, info = env.step(action)

# 新版本
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## 最佳实践

### 1. **选择合适的配置**
- 快速原型: 使用 `small` 配置
- 标准训练: 使用 `standard` 配置
- 精细调优: 使用 `dense` 配置

### 2. **状态表示选择**
- 简单任务: 使用标准状态
- 复杂任务: 使用高级状态

### 3. **奖励类型选择**
- 明确目标: 使用稀疏奖励
- 学习过程: 使用密集奖励
- 平衡训练: 使用标准奖励

### 4. **并行环境使用**
- CPU训练: 4-8个环境
- GPU训练: 8-16个环境
- 内存限制: 减少环境数量

## 故障排除

### 常见问题

1. **维度不匹配**: 检查状态表示配置
2. **奖励异常**: 检查奖励类型设置
3. **性能问题**: 调整并行环境数量
4. **内存不足**: 减少批次大小或环境数量

### 调试技巧

1. 使用测试脚本验证环境
2. 检查环境信息输出
3. 监控训练统计信息
4. 使用渲染功能可视化

## 总结

新的规范化环境提供了：
- 更好的模块化设计
- 更丰富的功能特性
- 更标准化的接口
- 更灵活的配置选项
- 更完善的错误处理

这些改进使得五子棋环境更适合强化学习研究和应用，同时保持了向后兼容性。
