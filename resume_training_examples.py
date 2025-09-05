"""
恢复训练使用示例
"""
import os
import subprocess
import sys


def run_command(cmd):
    """运行命令并显示输出"""
    print(f"执行命令: {cmd}")
    print("-" * 50)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    print("-" * 50)
    return result.returncode == 0


def main():
    """主函数"""
    print("五子棋AI恢复训练示例")
    print("=" * 60)
    
    # 检查是否存在模型文件
    models_dir = "./models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            print(f"找到 {len(model_files)} 个模型文件:")
            for f in model_files:
                print(f"  - {f}")
        else:
            print("未找到模型文件")
    else:
        print("模型目录不存在")
    
    print("\n恢复训练的方法:")
    print("=" * 60)
    
    # 方法1: 指定具体模型文件
    print("1. 指定具体模型文件恢复训练:")
    print("   python train.py --resume ./models/ppo_gomoku_episode_1000.pth")
    
    # 方法2: 自动恢复最新检查点
    print("\n2. 自动恢复最新检查点:")
    print("   python train.py --auto-resume")
    
    # 方法3: 使用并行环境恢复训练
    print("\n3. 使用并行环境恢复训练:")
    print("   python train.py --auto-resume --num-envs 8 --use-vectorized")
    
    # 方法4: 调整参数恢复训练
    print("\n4. 调整参数恢复训练:")
    print("   python train.py --auto-resume --num-episodes 20000 --lr 1e-4")
    
    print("\n参数说明:")
    print("=" * 60)
    print("--resume <path>        : 指定模型文件路径")
    print("--auto-resume          : 自动恢复最新检查点")
    print("--num-envs <n>         : 并行环境数量")
    print("--use-vectorized       : 使用向量化环境")
    print("--num-episodes <n>     : 总训练轮数")
    print("--lr <rate>            : 学习率")
    print("--save-interval <n>    : 保存间隔")
    
    print("\n实际运行示例:")
    print("=" * 60)
    
    # 询问用户是否要运行示例
    choice = input("是否运行自动恢复训练示例? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("运行自动恢复训练示例...")
        success = run_command("python train.py --auto-resume --num-episodes 100 --num-envs 4 --save-interval 50")
        
        if success:
            print("示例运行成功!")
        else:
            print("示例运行失败，请检查错误信息")
    else:
        print("跳过示例运行")
    
    print("\n注意事项:")
    print("=" * 60)
    print("1. 确保模型文件存在且完整")
    print("2. 恢复训练时会自动加载训练统计信息")
    print("3. 系统会自动清理旧检查点，只保留最新的3个")
    print("4. 可以使用不同的参数继续训练")
    print("5. 建议定期备份重要的模型文件")


if __name__ == "__main__":
    main()
