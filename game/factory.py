"""
五子棋环境工厂
"""
from typing import Dict, Any, Optional
from .environment import GomokuEnvironment
from .config import GomokuConfig, STANDARD_CONFIG, SPARSE_CONFIG, DENSE_CONFIG, SMALL_BOARD_CONFIG, LARGE_BOARD_CONFIG


class GomokuEnvironmentFactory:
    """五子棋环境工厂类"""
    
    # 预定义配置
    CONFIGS = {
        'standard': STANDARD_CONFIG,
        'sparse': SPARSE_CONFIG,
        'dense': DENSE_CONFIG,
        'small': SMALL_BOARD_CONFIG,
        'large': LARGE_BOARD_CONFIG
    }
    
    @classmethod
    def create(cls, 
               config_name: Optional[str] = None,
               config: Optional[GomokuConfig] = None,
               **kwargs) -> GomokuEnvironment:
        """
        创建五子棋环境
        
        Args:
            config_name: 预定义配置名称
            config: 自定义配置对象
            **kwargs: 额外参数，会覆盖配置中的参数
            
        Returns:
            五子棋环境实例
        """
        if config is not None:
            # 使用提供的配置
            env_config = config
        elif config_name is not None and config_name in cls.CONFIGS:
            # 使用预定义配置
            env_config = cls.CONFIGS[config_name]
        else:
            # 使用默认配置
            env_config = STANDARD_CONFIG
        
        # 创建配置字典
        config_dict = env_config.to_dict()
        
        # 用kwargs覆盖配置
        config_dict.update(kwargs)
        
        # 创建环境
        return GomokuEnvironment(**config_dict)
    
    @classmethod
    def create_standard(cls, **kwargs) -> GomokuEnvironment:
        """创建标准环境"""
        return cls.create('standard', **kwargs)
    
    @classmethod
    def create_sparse(cls, **kwargs) -> GomokuEnvironment:
        """创建稀疏奖励环境"""
        return cls.create('sparse', **kwargs)
    
    @classmethod
    def create_dense(cls, **kwargs) -> GomokuEnvironment:
        """创建密集奖励环境"""
        return cls.create('dense', **kwargs)
    
    @classmethod
    def create_small(cls, **kwargs) -> GomokuEnvironment:
        """创建小棋盘环境"""
        return cls.create('small', **kwargs)
    
    @classmethod
    def create_large(cls, **kwargs) -> GomokuEnvironment:
        """创建大棋盘环境"""
        return cls.create('large', **kwargs)
    
    @classmethod
    def create_custom(cls, 
                     board_size: int = 15,
                     reward_type: str = 'standard',
                     **kwargs) -> GomokuEnvironment:
        """
        创建自定义环境
        
        Args:
            board_size: 棋盘大小
            reward_type: 奖励类型
            **kwargs: 其他参数
            
        Returns:
            五子棋环境实例
        """
        config = GomokuConfig(
            board_size=board_size,
            reward_type=reward_type,
            **kwargs
        )
        return cls.create(config=config)
    
    @classmethod
    def list_configs(cls) -> Dict[str, str]:
        """
        列出所有预定义配置
        
        Returns:
            配置名称和描述的字典
        """
        return {
            'standard': '标准配置：15x15棋盘，标准奖励',
            'sparse': '稀疏奖励配置：只在游戏结束时给予奖励',
            'dense': '密集奖励配置：包含位置价值和威胁检测',
            'small': '小棋盘配置：9x9棋盘，适合快速训练',
            'large': '大棋盘配置：19x19棋盘，更复杂的游戏'
        }
    
    @classmethod
    def get_config(cls, config_name: str) -> Optional[GomokuConfig]:
        """
        获取预定义配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置对象，如果不存在返回None
        """
        return cls.CONFIGS.get(config_name)


# 便捷函数
def create_gomoku_env(config_name: str = 'standard', **kwargs) -> GomokuEnvironment:
    """
    创建五子棋环境的便捷函数
    
    Args:
        config_name: 配置名称
        **kwargs: 额外参数
        
    Returns:
        五子棋环境实例
    """
    return GomokuEnvironmentFactory.create(config_name, **kwargs)


def create_standard_env(**kwargs) -> GomokuEnvironment:
    """创建标准环境的便捷函数"""
    return GomokuEnvironmentFactory.create_standard(**kwargs)


def create_sparse_env(**kwargs) -> GomokuEnvironment:
    """创建稀疏奖励环境的便捷函数"""
    return GomokuEnvironmentFactory.create_sparse(**kwargs)


def create_dense_env(**kwargs) -> GomokuEnvironment:
    """创建密集奖励环境的便捷函数"""
    return GomokuEnvironmentFactory.create_dense(**kwargs)
