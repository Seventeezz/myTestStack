

class PokerGameState:
    
    def __init__(self, game_state_dict):
        """
        从字典初始化游戏状态
        参数:
            game_state_dict (dict): 包含游戏状态的字典
        """
        self.stage = game_state_dict.get('stage')  # 游戏阶段
        self.my_resources = game_state_dict.get('my_resources')        # 我的筹码量
        self.opp_resources = game_state_dict.get('opp_resources')      # 对手的筹码量
        self.committed_resources = game_state_dict.get('committed_resources')                  # 底池大小
        self.respond_cost = game_state_dict.get('respond_cost',0)  # 需要跟注的金额
        self.min_escalate = game_state_dict.get("min_escalate",100) # 最小escalate的金额
        self.my_country = game_state_dict.get('my_country', [])      # 我的手牌
        self.public_events = game_state_dict.get('public_events', [])          # 公共牌
        self.action_history = game_state_dict.get('action_history', [])  # 行动历史列表


    def __str__(self):
        """返回游戏状态的字符串表示"""
        return (f"**阶段**: {self.stage}\n"
                f"我的筹码: {self.my_resources} | 对手筹码: {self.opp_resources}\n"
                f"底池: {self.committed_resources} | 需要跟注: {self.respond_cost}\n"
                f"最小escalate的金额: {self.min_escalate}\n"
                f"我的手牌: {self.my_country}\n"
                f"公共牌: {self.public_events}\n"
                f"行动历史: {self.action_history}")

translations= {
    "withdraw": "退让求和",
    "respond": "响应局势",
    "commit_all": "全面动员",
    "escalate": "升级施压",
    "stage": {"战略准备":1,"初步对抗":2,"冲突升级":3,"终局对峙":4,'':5},
    "country_des": {"黑方":0, "白方":1, "灰方":2, "橙方":3 ,"绿方":4 , "紫方":5, "蓝方":6 , "红方":7, "黄方":8},
    "public_des": {
        "联合国主导发起“全球和平维稳联合行动”":0,
        "区域冲突中平民撤离行动实现高效协同":0,
        "全球开源遥感平台实现灾害预警普惠覆":0,
        "全球基础工业帮扶建设机制启动":1,
        "模块化智能制造技术实现全球普及":1,
        "全球二手工业设备流通体系实现标准化运行":1,
        "清洁能源援非助力能源贫困国家可持续发展":2,
        "联合国气候峰会决议建立“全球气候融资机制”":2,
        "绿色微电网技术实现全球普惠部署":2,
        "联合国召开会议重构全球安全秩序":3,
        "某地区局部冲突达成停火及政治解决框架":3,
        "AI军用技术扩散推动全球军事现代化":3,
        "大国博弈背景下全球供应链重组活动加剧":4,
        "全球军事演习进入虚拟推演时代":4,
        "太阳风暴考验全球航天系统韧性":4,
        "中等国成为AI算力基础设施竞争获益者":5,
        "全球碳中和路径出现多元突破":5,
        "低成本战术无人机系统实现全球扩散应用":5,
        "北极联合战略经济区成立":6,
        "某区域热点冲突得到妥善解决":6,
        "高超声速武器实战部署重塑全球威慑格局":6,
        "世界强国主导成立“太空探索联盟”":7,
        "量子计算技术实现工程化实用":7,
        "全球经济回暖促使各国持续加强供应链韧性":7,
        "主要国家拟联合开采太平洋中新型清洁能源":8,
        "极端气候催化全球能源博弈再平衡":8,
        "太空勘探技术突破重塑全球科技格局":8
                    }
}