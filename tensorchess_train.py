#!/usr/bin/python
import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from   tensorflow.python.keras import layers, models
from   tensorflow.python.keras.models import load_model

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

def generate_training_data(pgn_file):
    states, moves = [], []
    f = open(pgn_file)
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            print("Game doesn't exist")
            break

        board = game.board()
        for move in game.mainline_moves():
            states.append(board_to_input(board))
            moves.append(move.from_square * 64 + move.to_square)
            board.push(move)

    return np.array(states), np.array(moves)

def crate_policy_network():
    input_layer = layers.Input(shape=(8,8,12))

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(64 * 64, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

states, moves = generate_training_data("pgn/kasparov-deep-blue-1997.pgn")

model = None
model_file = "chess_model.tf"

if not os.path.exists(model_file):
    model = crate_policy_network()
    model.fit(states, moves, epochs=10, batch_size=32, validation_split=0.2)
else:
    model = load_model(model_file)