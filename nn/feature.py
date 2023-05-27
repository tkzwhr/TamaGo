"""ニューラルネットワークの入力特徴生成処理
"""
import numpy as np

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone
from common.print_console import print_err


def generate_input_planes(board: GoBoard, color: Stone, sym: int=0) -> np.ndarray:
    """ニューラルネットワークの入力データを生成する。

    Args:
        board (GoBoard): 碁盤の情報。
        color (Stone): 手番の色。
        sym (int, optional): 対称形の指定. Defaults to 0.

    Returns:
        numpy.ndarray: ニューラルネットワークの入力データ。
    """
    board_data = board.get_board_data(sym)
    board_size = board.get_board_size()
    # 手番が白の時は石の色を反転する.
    if color is Stone.WHITE:
        board_data = [datum if datum == 0 else (3 - datum) for datum in board_data]

    # print_err("===")
    # print_err(np.array(board_data).reshape(board_size, board_size))
    # print_err("---")

    # 碁盤の各交点の状態
    #     空点 : 1枚目の入力面
    #     自分の石 : 2枚目の入力面
    #     相手の石 : 3枚目の入力面
    board_plane = np.identity(3)[board_data].transpose()

    # 直前の着手を取得
    _, previous_move, _ = board.record.get(board.moves - 1)

    # 直前の着手の座標
    #     着手 : 4枚目の入力面
    #     パス : 5枚目の入力面
    if board.moves > 1 and previous_move == PASS:
        history_plane = np.zeros(shape=(1, board_size ** 2))
        pass_plane = np.ones(shape=(1, board_size ** 2))
    else:
        previous_move_data = [1 if previous_move == board.get_symmetrical_coordinate(pos, sym) \
            else 0 for pos in board.onboard_pos]
        history_plane = np.array(previous_move_data).reshape(1, board_size**2)
        pass_plane = np.zeros(shape=(1, board_size ** 2))

    # 手番の色 (6番目の入力面)
    # 黒番は1、白番は-1
    color_plane = np.ones(shape=(1, board_size**2))
    if color == Stone.WHITE:
        color_plane = color_plane * -1
    
    # input_data = np.concatenate([board_plane, history_plane, pass_plane, color_plane]) \
    # .reshape(6, board_size, board_size).astype(np.float32) # pylint: disable=E1121

    # ここから新しい入力特徴

    # 自分と相手の空きダメの数 (7～12番目の入力面)
    #   4以上は安全
    #   3は要注意
    #   2以下は危険
    def align_lib(board: GoBoard, pos: int) -> int:
        num = board.strings.get_num_liberties(pos)
        if num == 1:
            return 2
        elif num > 4:
            return 4
        else:
            return num
        
    lib_array = [align_lib(board, pos) for pos in board.onboard_pos]

    my_lib_array = [int(lib_array[i] * board_plane[1][i]) for i in range(board_size**2)]
    my_lib_plane = (np.identity(5)[my_lib_array]).transpose()[2:5]
    opponent_lib_array = [int(lib_array[i] * board_plane[2][i]) for i in range(board_size**2)]
    opponent_lib_plane = (np.identity(5)[opponent_lib_array]).transpose()[2:5]

    # 単独の石にツケてきた場合、2目にナラビの場合は気にしたほうが良い

    previous_move_neighbors = board.strings.get_neighbor4(previous_move)

    def size(board: GoBoard, pos: int):
        string = board.strings.string[board.strings.get_id(pos)]
        if string.exist():
            return string.get_size()
        else:
            return 0
    def libs(board: GoBoard, pos: int):
        string = board.strings.string[board.strings.get_id(pos)]
        if string.exist():
            return string.get_num_liberties()
        else:
            return 0

    previous_move_size = [previous_move, size(board, previous_move), libs(board, previous_move)]
    previous_move_neighbors_sizes = [[pos, size(board, pos), libs(board, pos)] for pos in previous_move_neighbors]
    previous_move_neighbors_sizes.append(previous_move_size)
    
    # 単独の石にツケてきた

    previous_move_singles = [data[0] for data in previous_move_neighbors_sizes if data[1] == 1 and data[2] == 3]
    if len(previous_move_singles) == 2:
        previous_move_single_plane = np.array([1 if board.get_symmetrical_coordinate(pos, sym) in previous_move_singles \
            else 0 for pos in board.onboard_pos]).reshape(1, board_size**2)
    else:
        previous_move_single_plane = np.zeros(shape=(1, board_size ** 2))

    # 2目にナラビ

    previous_move_two_lines = [data[0] for data in previous_move_neighbors_sizes if data[1] == 2 and data[2] <= 4]
    if len(previous_move_two_lines) == 3:
        stones2d = [board.strings.get_stone_coordinates(board.strings.get_id(pos)) for pos in previous_move_two_lines]
        stones = [x for row in stones2d for x in row]
        previous_move_two_lines_plane = np.array([1 if board.get_symmetrical_coordinate(pos, sym) in stones \
            else 0 for pos in board.onboard_pos]).reshape(1, board_size**2)
    else:
        previous_move_two_lines_plane = np.zeros(shape=(1, board_size ** 2))

    input_data = np.concatenate([
        board_plane,
        history_plane,
        pass_plane,
        color_plane,
        my_lib_plane,
        opponent_lib_plane,
        previous_move_single_plane,
        previous_move_two_lines_plane
    ]).reshape(14, board_size, board_size).astype(np.float32) # pylint: disable=E1121

    return input_data


def generate_target_data(board:GoBoard, target_pos: int, sym: int=0) -> np.ndarray:
    """教師あり学習で使用するターゲットデータを生成する。

    Args:
        board (GoBoard): 碁盤の情報。
        target_pos (int): 教師データの着手の座標。
        sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

    Returns:
        np.ndarray: Policyのターゲットラベル。
    """
    target = [1 if target_pos == board.get_symmetrical_coordinate(pos, sym) else 0 \
        for pos in board.onboard_pos]
    # パスだけ対称形から外れた末尾に挿入する。
    target.append(1 if target_pos == PASS else 0)
    #target_index = np.where(np.array(target) > 0)
    #return target_index[0]
    return np.array(target)


def generate_rl_target_data(board: GoBoard, improved_policy_data: str, sym: int=0) -> np.ndarray:
    """Gumbel AlphaZero方式の強化学習で使用するターゲットデータを精鋭する。

    Args:
        board (GoBoard): 碁盤の情報。
        improved_policy_data (str): Improved Policyのデータをまとめた文字列。
        sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

    Returns:
        np.ndarray: Policyのターゲットデータ。
    """
    split_data = improved_policy_data.split(" ")[1:]
    target_data = [1e-18] * len(board.board)

    for datum in split_data[1:]:
        pos, target = datum.split(":")
        coord = board.coordinate.convert_from_gtp_format(pos)
        target_data[coord] = float(target)

    target = [target_data[board.get_symmetrical_coordinate(pos, sym)] for pos in board.onboard_pos]
    target.append(target_data[PASS])

    return np.array(target)
