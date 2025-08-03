import time
import random
import os
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

from GUI.poker import DoylesGame
from GUI.test_bot import TestBot
from GUI.logger import Logger
from Player.continual_resolving import ContinualResolving
from GUI.client import client as browser



# pystack = TestBot()
pystack = ContinualResolving()
logger = Logger('data/logs.csv')
game = DoylesGame(bot=pystack, logger=logger)
GAME_IS_RUNNING = False

app = Flask(__name__, static_folder='../client', static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    always_connect=True,
    transports=['polling', 'websocket']
)
CORS(app, supports_credentials=True)


# app = Flask(__name__, static_folder='../client', static_url_path='')
# # 配置 CORS
# CORS(app, resources={
#     r"/*": {
#         "origins": "*",
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True
#     }
# })

# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, 
#     cors_allowed_origins="*",
#     async_mode='threading',
#     logger=True,
#     engineio_logger=True,
#     ping_timeout=60,
#     ping_interval=25,
#     max_http_buffer_size=1e8,
#     always_connect=True,
#     transports=['polling']  # 只使用 polling 传输
# )

@app.route('/')
def index():
    try:
        return send_from_directory(app.static_folder, 'game.html')
    except Exception as e:
        print(f"Error serving game.html: {str(e)}")
        return "Error loading game", 500

@socketio.on('connect')
def test_connect():
    print('------USER CONNECTED------')
    return {'status': 'connected'}

@socketio.on('start_game')
def start_game():
    try:
        print('------STARTING GAME------')
        avg_wins = logger.get_avg_wins()
        browser.change_stats(avg_wins=avg_wins)
        starting_player = 'player' if random.random() > 0.5 else 'bot'
        print('starting_player:', starting_player)
        time.sleep(2)
        game.start_round(starting_player)
        global GAME_IS_RUNNING
        if GAME_IS_RUNNING:
            for _ in range(100):
                print('ERROR: more then one user are trying to connect or multiple tabs opened!')
        GAME_IS_RUNNING = True
        return {'code': 'success'}
    except Exception as e:
        print(f"Error in start_game: {str(e)}")
        return {'code': 'error', 'message': str(e)}

@socketio.on('player_send_action')
def player_send_action(action, amount):
    # try:
    print('---- PLAYER ACTION: {} {} ----'.format(action, amount))
    if game.current_player == 'player':
        success, action, amount = game.player_action(action, amount)
        return {'code': 'success', 'action': action, 'amount': amount}
    else:
        return {'code': 'not_your_turn'}
    # except Exception as e:
    #     print(f"Error in player_send_action: {str(e)}")
    #     return {'code': 'error', 'message': str(e)}

@socketio.on('player_received_end_game_msg')
def player_received_end_game_msg():
    try:
        print('------RESETING GAME------')
        global GAME_IS_RUNNING
        GAME_IS_RUNNING = False
        return {'code': 'success'}
    except Exception as e:
        print(f"Error in player_received_end_game_msg: {str(e)}")
        return {'code': 'error', 'message': str(e)}

def run_server():
    try:
        socketio.run(app, 
                    host='0.0.0.0',
                    port=8000, 
                    debug=False,
                    use_reloader=False,
                    log_output=True,
                    allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
