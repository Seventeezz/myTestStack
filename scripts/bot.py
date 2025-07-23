import random
import sys 
import os 
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict
from fastapi import FastAPI
import uvicorn
from uvicorn.config import LOGGING_CONFIG
import argparse
from Player.continual_resolving import ContinualResolving
from Settings import PokerGameState,translations,arguments
import numpy as np

class BaselineBot:
    def __init__(self,is_debug = True):
        self.app = FastAPI(debug = is_debug)
        
        # 注册API端点
        self.app.post("/new_hand")(self.new_hand)
        self.app.post("/act")(self.act)
        self.app.post("/result")(self.result)
        self.bot = ContinualResolving()
        self.poker_game_state = None
        self.current_action = {"action":"", "amount":-1}
    
    def new_hand(self, payload: Dict):
        print("\n--------------- 新的一局开始 -----------------")
        self.bot.start_new_hand_for_race(my_country = payload["country"], player_is_small_blind = payload["first_to_action"])
        return {"status": "ack"}
    
    def act(self, payload: Dict):
        self.poker_game_state = PokerGameState(payload)
        stage = translations["stage"][self.poker_game_state.stage]
        my_country = translations["country_des"][self.poker_game_state.my_country]
        public_events = np.array([], dtype= np.int16)
        for event in self.poker_game_state.public_events:
            event_des = translations["public_des"][event[0]]
            public_events = np.append(public_events,event_des)
        self.current_action = self.bot.compute_action_for_race(
                public_events, 
                200 - self.poker_game_state.my_resources,
                200 - self.poker_game_state.opp_resources)
        print(self.current_action)
        # 将numpy类型转换为Python原生类型
        return {
            'action': self.current_action['action'],
            'amount': int(self.current_action['amount'])
        }

    def result(self, payload: Dict):
        print(f"[胜者]: {payload['winner']}")
        return {"status": "received"}

# 命令行启动参数如下:
# uvicorn baseline_bot:app --host 127.0.0.1 --port 5002
if __name__ == "__main__":
        # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="运行RandomStrategyBot应用")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001, help="服务器端口 (默认: 5001)")
    args = parser.parse_args()

    # 创建随机策略者实例
    bot = BaselineBot()
    app = bot.app
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=LOGGING_CONFIG
    )