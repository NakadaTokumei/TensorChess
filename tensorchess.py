#!/usr/bin/python
import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from   tensorflow.keras import layers, models
from   tensorflow.keras.models import load_model

class TensorModel:
    def __init__(self, model=models.Model()):
        self.model = model
        self.attempt = 0
        pass

    def create(self):
        input_layer = layers.Input(shape=(8,8,12))

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        for _ in range(4):
            skip = x
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.Add()([x, skip])

        x = layers.Flatten()(x)

        policy_output = layers.Dense(64 * 64, activation='softmax', name="policy_output")(x)
        value_output = layers.Dense(1, activation='tanh', name="value_output")(x)

        self.model = models.Model(inputs=input_layer, outputs=[policy_output, value_output])
        self.model.compile(
            optimizer='adam',
            loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
            metrics={'policy_output': 'accuracy', 'value_output': 'mse'}
        )
        pass

    def load(self, model_file):
        self.model = load_model(model_file)
        pass

    def board_to_input(self, board : chess.Board):
        board_matrix = np.zeros((8, 8, 12))
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_index = piece.piece_type - 1
            if piece.color == chess.BLACK:
                piece_index += 6
            board_matrix[row, col, piece_index] = 1

        return board_matrix
    
    def train(self, pgn_file):
        count = 0
        states, policies, values = [], [], []
        f = open(pgn_file)
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                print("Game doesn't exist")
                break

            board = game.board()
            game_states, game_policies = [], []

            for move in game.mainline_moves():
                game_states.append(self.board_to_input(board))
                policy = np.zeros(64 * 64)
                policy[move.from_square * 64 + move.to_square] = 1
                game_policies.append(policy)

                board.push(move)

            result = game.headers["Result"]
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0
            
            game_values = [reward] * len(game_states)

            states.extend(game_states)
            policies.extend(game_policies)
            values.extend(game_values)

            count += 1

            if count >= 1024:
                np_states = np.array(states)
                np_policies = np.array(policies)
                np_values = np.array(values)

                states = []
                policies = []
                values = []

                self.model.fit(
                np_states, 
                {"policy_output": np_policies, "value_output" : np_values },  
                epochs=10, 
                batch_size=32, 
                validation_split=0.2)

                count = 0

        np_states = np.array(states)
        np_policies = np.array(policies)
        np_values = np.array(values)

        self.model.fit(
            np_states, 
            {"policy_output": np_policies, "value_output" : np_values },  
            epochs=10, 
            batch_size=64, 
            validation_split=0.2)
        pass

    def save(self, model_file):
        self.model.save(model_file)

    def predict(self, input):
        return self.model.predict(input, verbose=0)

class MCTS_Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))
    
    def best_child(self, c_param=1.0):
        choice_weights = [
            (child.value_sum / (child.visits + 1e-8)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choice_weights)]
    
    def expand(self, move, policy_prob):
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTS_Node(new_board, parent=self)
        child_node.prior = policy_prob
        self.children[move] = child_node
        return child_node
    
    def backpropagate(self, result):
        self.visits += 1
        self.value_sum += result
        if self.parent:
            self.parent.backpropagate(result)


def board_to_input(board : chess.Board):
    board_matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_index = piece.piece_type - 1
        if piece.color == chess.BLACK:
            piece_index += 6
        board_matrix[row, col, piece_index] = 1

    return board_matrix

def get_best_move(board : chess.Board, model : models.Model):
    state = np.expand_dims(board_to_input(board), axis=0)
    move_probabilities = model.predict(state)[0]

    legal_moves = list(board.legal_moves)
    best_move = max(legal_moves, key=lambda move: move_probabilities[move.from_square * 64 + move.to_square])

    return best_move

def simulate(board, model):
    sim_board = board.copy()
    while not sim_board.is_game_over():
        move = get_best_move(sim_board, model)
        sim_board.push(move)
    
    result = sim_board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0
    
def mcts_with_policy(model, board, num_simulations=200, c_param=1.0):
    root = MCTS_Node(board)
    state = board_to_input(board)
    policy, _ = model.predict(np.expand_dims(state, axis=0))
    policy = policy[0]

    legal_moves = list(board.legal_moves)
    policy_probs = np.zeros(64 * 64)
    for move in legal_moves:
        move_idx = move.from_square * 64 + move.to_square
        policy_probs[move_idx] = policy[move_idx]

    for _ in range(num_simulations):
        node = root

        while node.is_fully_expanded() and len(node.children) > 0:
            node = node.best_child(c_param)

        if not node.is_fully_expanded():
            legal_moves = list(node.board.legal_moves)
            for move in legal_moves:
                move_idx = move.from_square * 64 + move.to_square
                if move not in node.children:
                    node.expand(move, policy_probs[move_idx])

        state = board_to_input(node.board)
        _, value = model.predict(np.expand_dims(state, axis=0))
        result = value[0][0]

        node.backpropagate(result)
    
    best_move = max(root.children.items(), key=lambda child: child[1].visits)[0]

    for move, child in root.children.items():
        move_idx = move.from_square * 64 + move.to_square
        policy_probs[move_idx] = child.visits / root.visits

    return best_move, policy_probs

    
def mcts(board, model, num_simulations=1000):
    root = MCTS_Node(board)

    for _ in range(num_simulations):
        node = root

        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_fully_expanded():
            node = node.expand()

        result = simulate(node.board, model)

        node.backprogate(result)
    
    return root.best_move()

model_file = "chess_model_v3.keras"

model = TensorModel()
model.load(model_file)

if not os.path.exists(model_file):
    print("Sorry but ai model not exist...")
    exit(-1)
else:
    model.load(model_file)

board = chess.Board()
def main():
    while not board.is_game_over():
        line = input().strip()
        print(board)

        if line == "uci":
            print("id name TensorChess")
            print("id author NakadaTokumei")
            print("uciok")

        elif line.startswith("position"):
            tokens = line.split()
            if "startpos" in tokens:
                board.reset()
                if "moves" in tokens:
                    moves_start_idx = tokens.index("moves") + 1
                    moves = tokens[moves_start_idx:]
                    for move in moves:
                        board.push_uci(move)
            elif "fen" in tokens:
                fen_start_idx = tokens.index("fen") + 1
                fen = " ".join(tokens[fen_start_idx:fen_start_idx + 6])
                board.set_fen(fen)
        
        elif line.startswith("go"):
            ai_move, _ = mcts_with_policy(model, board)
            print(f"bestmove {ai_move}")
        
        elif line == "isready":
            print("readyok")
        
        elif line == "quit":
            break

        # if board.turn == chess.WHITE:
        #     user_move = input("Move: ")
        #     board.push_uci(user_move)
        # else:
        #     ai_move = get_best_move(board, model)
        #     board.push(ai_move)
        #     print(f"AI moved: {ai_move}")
    
    model.train()

main()

print("Done...")
print("Result: ", board.result())