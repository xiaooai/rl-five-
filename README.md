# 五子棋AI - PPO算法训练

这是一个使用PPO（Proximal Policy Optimization）算法训练五子棋AI的项目。项目从零开始构建，包含完整的游戏环境、神经网络模型、PPO算法实现以及训练和评估工具。

## 项目结构

```
five-cheese/
├── requirements.txt          # 依赖包
├── README.md                # 项目说明
├── game/                    # 游戏环境
│   ├── __init__.py
│   ├── board.py            # 棋盘逻辑
│   └── environment.py      # 游戏环境
├── model/                   # 神经网络模型
│   ├── __init__.py
│   └── network.py          # PPO网络结构
├── ppo/                     # PPO算法
│   ├── __init__.py
│   ├── agent.py            # PPO智能体
│   └── memory.py           # 经验回放
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
└── play.py                  # 人机对战
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

#### 基本训练
```bash
python train.py
```

#### 并行环境训练
```bash
# 使用4个并行环境
python train.py --num-envs 4

# 使用向量化环境
python train.py --num-envs 8 --use-vectorized
```

#### 恢复训练
```bash
# 指定模型文件恢复
python train.py --resume ./models/ppo_gomoku_episode_1000.pth

# 自动恢复最新检查点
python train.py --auto-resume

# 使用并行环境恢复训练
python train.py --auto-resume --num-envs 8 --use-vectorized
```

#### 自定义参数训练
```bash
python train.py --num-episodes 5000 --board-size 15 --lr 3e-4 --device cuda
```

主要训练参数：
- `--num-episodes`: 训练轮数（默认10000）
- `--board-size`: 棋盘大小（默认15）
- `--num-envs`: 并行环境数量（默认16）
- `--use-vectorized`: 使用向量化环境
- `--lr`: 学习率（默认3e-4）
- `--num-steps`: 每轮收集步数（默认4096）
- `--batch-size`: 批次大小（默认2048）
- `--device`: 设备选择（auto/cpu/cuda）
- `--resume`: 恢复训练的模型路径
- `--auto-resume`: 自动恢复最新检查点

### 2. 评估模型

```bash
python evaluate.py --model-path ./models/ppo_gomoku_final.pth --num-games 100
```

评估参数：
- `--model-path`: 模型文件路径
- `--num-games`: 评估游戏数量（默认100）
- `--output-dir`: 结果输出目录（默认./evaluation）

### 3. 人机对战

```bash
python play.py --model-path ./models/ppo_gomoku_final.pth
```

对战参数：
- `--model-path`: 模型文件路径
- `--human-first`: 人类先手（可选）

## 算法说明

### PPO算法特点

本项目使用PPO算法训练五子棋AI，PPO是一种策略梯度方法，具有以下特点：

1. **策略裁剪**: 通过限制策略更新的幅度来保证训练的稳定性
2. **GAE优势估计**: 使用广义优势估计来减少方差
3. **Actor-Critic架构**: 同时学习策略和价值函数
4. **经验回放**: 使用经验缓冲区进行批量更新

### 网络架构

- **卷积层**: 3层卷积网络提取棋盘特征
- **全连接层**: 2层全连接网络进行特征融合
- **双头输出**: 策略头输出动作概率，价值头输出状态价值

### 训练策略

- **奖励设计**: 获胜+100，失败-100，平局0，每步+1
- **动作掩码**: 只允许在空位落子
- **经验收集**: 每轮收集2048步经验
- **批量更新**: 使用64的批次大小进行4轮更新

## 技术特点

- ✅ 使用PyTorch实现神经网络
- ✅ 支持GPU加速训练
- ✅ 并行环境训练支持（3-4倍速度提升）
- ✅ 向量化环境优化
- ✅ 可配置的训练参数
- ✅ 实时训练进度监控
- ✅ 支持模型保存和加载
- ✅ 自动检查点恢复功能
- ✅ 智能检查点管理
- ✅ 完整的评估和测试工具
- ✅ 人机对战功能
- ✅ 训练曲线可视化
- ✅ 详细的日志记录

## 训练建议

1. **硬件要求**: 建议使用GPU进行训练，CPU训练速度较慢
2. **并行环境**: 使用4-8个并行环境可以显著提升训练速度
3. **训练时间**: 完整训练需要数小时到数天，取决于硬件配置
4. **参数调优**: 可以调整学习率、批次大小等参数来优化训练效果
5. **监控训练**: 关注胜率、奖励等指标来判断训练效果
6. **检查点管理**: 定期保存检查点，支持中断后恢复训练
7. **内存管理**: 并行环境会占用更多内存，根据系统配置调整环境数量

## 文件说明

- `game/board.py`: 实现五子棋的基本规则和棋盘逻辑
- `game/environment.py`: 实现Gym风格的游戏环境
- `game/parallel_env.py`: 并行环境包装器实现
- `model/network.py`: 定义神经网络架构
- `ppo/agent.py`: 实现PPO算法核心逻辑
- `ppo/memory.py`: 实现经验回放缓冲区
- `train.py`: 主训练脚本（支持并行环境和恢复训练）
- `evaluate.py`: 模型评估脚本
- `play.py`: 人机对战脚本
- `test_parallel.py`: 并行环境测试脚本
- `test_resume.py`: 恢复训练功能测试脚本
- `resume_training_examples.py`: 恢复训练使用示例
- `PARALLEL_ENV_GUIDE.md`: 并行环境使用指南

## 注意事项

1. 确保安装了所有依赖包
2. 训练过程中会自动保存模型和日志
3. 评估时会生成详细的统计报告和可视化图表
4. 人机对战时输入格式为"行 列"（例如：7 7）
