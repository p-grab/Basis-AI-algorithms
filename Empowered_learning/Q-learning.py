import random
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, type):
        self.type = type
        self.position = None
        self.reward = 0
        self.q_table = None


class Board:
    def __init__(self, N=8, holes_prob=0.2) -> None:
        self.dimension = N
        self.points_list = self.initialize_points()
        self.terminate_points = self.init_terminate_points()
        self.start_point, self.end_point = self.start_end_points()
        self.holes = self.add_holes(holes_prob)
        self.moves = self.init_moves()
        self.rewards = self.init_rewards()

    def initialize_points(self):
        points_list = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                points_list.append((i, j))
        return points_list

    def start_end_points(self):
        while True:
            start_point = random.choice(self.points_list)
            if not self.terminate_points[start_point]:
                end_point = random.choice(self.points_list)
                if (
                    not self.terminate_points[end_point]
                    and (
                        end_point[0] != start_point[0] or start_point[1] != end_point[1]
                    )
                    and (
                        abs(end_point[0] - start_point[0]) > 3
                        or abs(start_point[1] - end_point[1]) > 3
                    )
                ):
                    break
        self.terminate_points[end_point] = True
        return start_point, end_point

    def init_terminate_points(self):
        terminate_points = {}
        for point in self.points_list:
            terminate_points[point] = False
        return terminate_points

    def print_board(self):
        out = ""
        for i in range(self.dimension):
            out += " | "

            for j in range(self.dimension):

                if self.start_point == (i, j):
                    out += "s"
                elif self.end_point == (i, j):
                    out += "e"

                elif (i, j) in self.holes:
                    out += "x"
                elif not self.terminate_points[i, j]:
                    out += " "
                else:
                    out += "-"

                out += " | "
            out += "\n\r"
        return out

    def print_board_points(self):
        out = ""
        for i in range(self.dimension):
            out += " | "
            for j in range(self.dimension):
                out += str(self.rewards[(i, j)])
                out += " | "
            out += "\n\r"
        return out

    def add_holes(self, holes_prob):
        non_terminate_points = [
            x for x in self.points_list if self.terminate_points[x] == False
        ]
        while True:
            holes = int(len(non_terminate_points) * holes_prob)
            hole_points = random.sample(non_terminate_points, holes)
            if (
                self.start_point not in hole_points
                and self.end_point not in hole_points
            ):
                break
        for hole_point in hole_points:
            self.terminate_points[hole_point] = True
        return hole_points

    def init_moves(self):
        legal_loves = {}
        for point in self.points_list:
            moves = []
            if point[0] > 0:
                moves.append((point[0] - 1, point[1]))
            if point[1] > 0:
                moves.append((point[0], point[1] - 1))
            if point[0] < self.dimension - 1:
                moves.append((point[0] + 1, point[1]))
            if point[1] < self.dimension - 1:
                moves.append((point[0], point[1] + 1))
            legal_loves[point] = moves
        return legal_loves

    def init_rewards(self):
        rewards = {}
        for point in self.points_list:
            if point == self.end_point:
                rewards[point] = 100
            elif point == self.start_point:
                rewards[point] = -1
            elif self.terminate_points[point] == True:
                rewards[point] = -100
            else:
                rewards[point] = -1
        return rewards

    def q_learning(
        self, agent, episodes=1000, epsilon=0.9, gamma=0.9, learning_rate=0.9, steps=50
    ):
        rewards_over_time = []
        agent.q_table = self.init_q_table()
        for i in range(episodes):
            agent.position = self.start_point
            agent.reward = 0
            step = 0
            while self.terminate_points[agent.position] is False and step < steps:
                old_position = agent.position
                agent.position = self.get_move(agent, epsilon)
                reward = self.rewards[agent.position]
                agent.reward += reward
                old_q_value = agent.q_table[old_position][agent.position]
                move_to_be = self.get_move(agent, 1)
                temporal_diff = reward + (
                    gamma * agent.q_table[agent.position][move_to_be] - old_q_value
                )
                new_q_value = old_q_value + (learning_rate * temporal_diff)
                agent.q_table[old_position][agent.position] = new_q_value
                step += 1

            rewards_over_time.append(agent.reward)
        return rewards_over_time

    def init_q_table(self):
        q_table = {}
        for move in self.moves:
            actions = {move: 0 for move in self.moves[move]}
            q_table[move] = actions
        return q_table

    def get_move(self, agent, epsilon):
        moves = self.moves[agent.position]
        if random.random() <= epsilon:
            moves_rewards = {}
            for move in moves:
                moves_rewards[move] = agent.q_table[agent.position][move]
            max_reward = max(moves_rewards.values())
            max_moves = [
                move
                for move in moves_rewards.keys()
                if moves_rewards[move] == max_reward
            ]
            return random.choice(max_moves)

        return random.choice(moves)

    def go_shortest_path(self, agent):
        path = []
        agent.position = self.start_point
        while self.terminate_points[agent.position] is False:
            path.append(agent.position)
            if agent.type == "ai":
                agent.position = self.get_move(agent, 1)
            else:
                agent.position = random.choice(self.moves[agent.position])
        path.append(agent.position)
        return path

    def print_shortest_path(self):
        out = ""
        for i in range(self.dimension):
            out += " | "

            for j in range(self.dimension):

                if self.start_point == (i, j):
                    out += "s"
                elif self.end_point == (i, j):
                    out += "e"

                elif (i, j) in self.holes:
                    out += "x"
                elif (i, j) in path:
                    out += "o"
                elif not self.terminate_points[i, j]:
                    out += " "
                else:
                    out += "-"

                out += " | "
            out += "\n\r"
        return out

    def check_board(self):
        visited = set()
        node = self.start_point
        self.dfs(visited, self.moves, node)

    def dfs(self, visited, moves, node):
        if node not in visited:
            visited.add(node)
            for neibour in self.moves[node]:
                self.dfs(visited, self.moves, node)

    def get_graph(self):
        graph = {}
        for point in self.moves.keys():
            moves = [
                move
                for move in self.moves[point]
                if self.terminate_points[move] is False or move == self.end_point
            ]
            graph[point] = moves
        return graph


def dfs(graph, start, end, visited=set()):
    if start == end:
        return True
    if start in visited or start not in graph:
        return False
    visited.add(start)
    for neighbor in graph[start]:
        if dfs(graph, neighbor, end, visited):
            return True
    return False


if __name__ == "__main__":
    while True:
        board = Board(8, 0.3)
        print(board.print_board())
        print(board.print_board_points())
        graph = board.get_graph()
        possible = dfs(graph, board.start_point, board.end_point)
        if possible:
            break
        else:
            print("No path between points")
    # print(board.moves)
    print("\n\n")
    # print(board.init_q_table())
    ai_agent = Agent("ai")
    random_agent = Agent("random")
    rewards_over_time = board.q_learning(
        ai_agent, episodes=500, epsilon=0.9, learning_rate=0.9, gamma=0.9, steps=50
    )
    path = board.go_shortest_path(ai_agent)
    # path = board.go_shortest_path(random_agent)
    print("Optimal path:")
    print(path)
    print("Path on board:")
    print(board.print_shortest_path())

    # x_axis = [x for x in range(len(rewards_over_time))]
    # plt.plot(x_axis, rewards_over_time, "o")
    best_reward = []
    best_score = -150
    for score in rewards_over_time:
        if score > best_score:
            best_reward.append(score)
            best_score = score
        else:
            best_reward.append(best_score)

    x_axis = [x for x in range(len(best_reward))]
    plt.plot(x_axis, best_reward)
    plt.xlabel("episodes")
    plt.ylabel("score")
    plt.title("Result over episodes")
    plt.show()
