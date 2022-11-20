import random
import copy
import time

TIMES = []
TURNS = []


class Player:
     def __init__(self, is_active, name):
        self.place = None
        self.is_active = is_active
        self.name = name


class Board:
    def __init__(self, player1, player2, width, height):
        self.width = width
        self.height = height
        self.move_count = 0
        self.player1 = player1
        self.player2 = player2

        self.board_list = []
        for i in range(width):
            for j in range(height):
                self.board_list.append((i,j))
        self.board_empty_list = {}
        for place in self.board_list:
            self.board_empty_list[place] = True


    def initial_players_pos(self):
        x = random.randint(0, self.width)
        y = random.randint(0, self.height)
        self.player1.place = (x,y)
        self.board_empty_list[(x,y)] = False
        while(True):
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            if x2 != x or y2 != y:
                break
        self.player2.place = (x2,y2)
        self.board_empty_list[(x2,y2)] = False


    def legal_moves(self, player):
        moves = []
        pos = player.place
        for i in range(-1,2):
            for j in range(-1,2):
                if ((0 <= pos[0] + i and pos[0] + i < self.width) and (0 <=  pos[1] + j and pos[1] + j < self.height)
                    and (self.board_empty_list[(pos[0] + i, pos[1] + j)])):
                        moves.append((pos[0] + i, pos[1] + j))
        return moves


    def get_active_player(self):
        if self.player1.is_active:
            return self.player1
        elif self.player2.is_active:
            return self.player2
        else:
            raise AttributeError('No active player in game!')


    def get_inactive_player(self):
        if self.get_active_player() == self.player1:
            return self.player2
        return self.player1


    def make_move(self, move):
        player = self.get_active_player()
        player.place = move
        self.board_empty_list[move] = False
        player_next = self.get_inactive_player()
        player.is_active = False
        player_next.is_active = True
        self.move_count += 1


    def board_to_str(self):
        out = ''
        for i in range(self.width):
            out += ' | '

            for j in range(self.height):

                if self.board_empty_list[i,j]:
                    out += ' '
                elif self.player1.place == (i,j) and self.board_empty_list[(i,j)] == False:
                    out += '1'
                elif self.player2.place == (i,j) and self.board_empty_list[(i,j)] == False:
                    out += '2'
                else:
                    out += '-'

                out += ' | '
            out += '\n\r'
        return out


    def lose_player(self):
        if len(self.legal_moves(self.get_active_player())) == 0:
            return self.get_active_player()


    def to_string_in_searching(self):
        out = ''
        for i in range(self.width):
            out += ' | '

            for j in range(self.height):

                if self.board_empty_list[i,j]:
                    out += ' '
                elif self.get_active_player().place == (i,j) and self.board_empty_list[(i,j)] == False:
                    if self.get_active_player().name == 'player1':
                        out += '1'
                    else:
                        out += '2'
                elif self.get_inactive_player().place == (i,j) and self.board_empty_list[(i,j)] == False:
                    if self.get_inactive_player().name == 'player1':
                        out += '1'
                    else:
                        out += '2'
                else:
                    out += '-'

                out += ' | '
            out += '\n\r'
        return out


def play(board_size, depth):
    p1 = Player(True, 'player1')
    p2 = Player(False, 'player2')
    board = Board(p1,p2, board_size, board_size)
    board.initial_players_pos()
    while True:
        #print(board.board_to_str())
        moves = board.legal_moves(board.get_active_player())
        if len(moves) == 0:
            # print(f"Game ends in {board.move_count} turns!")
            # print(f'Player {board.get_inactive_player().name} wins!')
            break
        if board.get_active_player().name  == 'player1':
            start = time.process_time()
            move = min_max_alg(board, 0, depth, p1.name)
            stop = time.process_time()
            TIMES.append(stop - start)
            #move = moves[random.randint(0, len(moves) -1)]
        else:
            move = moves[random.randint(0, len(moves) -1)]
            # start = time.process_time()
            # move = min_max_alg(board, 0, depth, p2.name)
            # stop = time.process_time()
            # TIMES.append(stop - start)
        board.make_move(move)
    TURNS.append(board.move_count)
    return board.get_inactive_player().name


def min_max_alg(board, depth, max_depth, who):
    # print(f'DEPTH:\t{depth}\ton move:\t{board.get_active_player().name}')
    # print(board.to_string_in_searching())
    if board.lose_player():
        if board.get_active_player().name != who:
            return 1000
        else:
            return -1000
    if (depth == max_depth):
        return len(board.legal_moves(board.get_active_player()))
    successors = []
    for move in board.legal_moves(board.get_active_player()):
        successors.append(move)
    output = {}
    for move in successors:
        copy_p1 = Player(True, board.get_active_player().name)
        copy_p1.place = board.get_active_player().place
        copy_p2 = Player(False, board.get_inactive_player().name)
        copy_p2.place = board.get_inactive_player().place
        copy_board = Board(copy_p1,copy_p2, board.width, board.height) 
        copy_board.board_empty_list = board.board_empty_list.copy()
        copy_board.make_move(move)

        output[move] = min_max_alg(copy_board, depth + 1, max_depth, who)

    if board.get_active_player().name == who:
        maximal = -1000
        for move in output.keys():
            if output[move] >= maximal:
                maximal = output[move]
        max_list = []
        for move in output.keys():
            if output[move] == maximal:
                max_list.append(move)
        return maximal if depth != 0 else random.choice(max_list)
    else:
        minimal = 1000
        for move in output.keys():
            if output[move] <= minimal:
                minimal = output[move]
        min_list = []
        for move in output.keys():
            if output[move] == minimal:
                min_list.append(move)
        return minimal if depth != 0 else random.choice(min_list)



#INICIAL PARAMETERS
board_size = 4
depth = 4
for d in range(1, depth + 1):
    TIMES = []
    TURNS = []
    win_p1 = 0
    win_p2 = 0
    for i in range(100):
        # print(i)
        winner = play(board_size=board_size, depth=d)
        if winner == 'player1':
            win_p1 +=1
        else:
            win_p2 +=1

    sum_times = 0
    sum_turns = 0
    for tim,turn in zip(TIMES, TURNS):
        sum_times += tim
        sum_turns += turn

    print(f'In {i + 1} games:\nplayer1 {win_p1} wins\nplayer2 {win_p2} wins')
    print(f'Average time for minimax with depth = {d}:\t {round(sum_times / len(TIMES), 5)} [seconds]')
    print(f'Average turns for minimax with depth = {d}:\t {round(sum_turns / len(TURNS), 1)}')
