import copy
import random
import board
import numpy as np
from Random_Game import Random_Game
import math
import time

timeout = 2
window_length = 3
def count_remaining(board, color='.'):
    count = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == color:
                count += 1
    return count
class statmodel:
    def __init__(self, window_length=3, fixed=False):
        self.mean = 0.5
        self.std = 0.5
        self.distribution = 'normal'
        self.window_length = window_length
        self.record = []
        self.t = 0
        self.time_decay = False
        self.fixed = fixed
    
    def prediction(self, predict_length=1):
        '''
        Create a max distribution based on model and sample from it.
        '''
        value = self.std * np.sqrt(2 * np.log(predict_length))
        # print(value)
        try:
            if len(value) > 0:
                return value[0]
        except:
            return value
        
    def getReward(self, value):
        """
        update mean/variance according to input value.
        """
        self.record.append(value)
        # self.mean = max(self.record[-self.window_length:])
        # self.mean = np.mean(self.record[-10:])
        if len(self.record) > self.window_length:
            if not self.fixed:
                mean = np.mean(self.record[-self.window_length:])
                std = np.std(self.record[-self.window_length:])
                self.std = 0.05 * std + 0.95 * self.std
                self.mean = 0.05 * value + 0.95 * self.mean
            else:
                self.std = np.std(self.record)
                self.mean = np.mean(self.record)
        #elif len(self.record) == self.window_length:
        else:
            self.std = np.std(self.record)
            self.mean = np.mean(self.record)
        # else:
        #     self.std = 0
        #     self.mean = 0.5

# Node类是状态，代表一种局势
class Node(object):
    def __init__(self):
        # 上一步棋对应的局势
        self.parent = None
        # 下一步棋对应的局势，以{sub_node:action}形式存储
        self.children = {}
        # 在MCTS随机搜索过程中，被访问过的次数
        self.visit_times = 0
        # 这种局势的评分
        self.quality_value = 0.0
        self.quality_value_2 = 0.0
        # 当前轮数。第一手为1
        self.round_index = 0
        # 初始化棋盘
        self.board_ = [['.' for _ in range(8)] for _ in range(8)]
        self.board_[3][4] = 'X'  # 黑棋棋子
        self.board_[4][3] = 'X'  # 黑棋棋子
        self.board_[3][3], self.board_[4][4] = 'O', 'O'  # 白棋棋子
        # 在该局面下，将要下棋一方的颜色，'X'为黑，'O'为白
        self.color = 'X'
        self.model = statmodel()

    # 获取对手颜色
    def oppsite_color(self,color):
        if color=='X':
            return 'O'
        else:
            return 'X'

    # 根据Node获得Board类的实例board
    def get_formate_board(self):
        formate_board = board.Board()
        for i in range(8):
            for j in range(8):
                formate_board._board[i][j] = self.board_[i][j]
        return formate_board

    # 获取当前局势下，可以做的action
    def get_valid_actions(self,color):
        # Board类提供了相关算法，所以先转化到Board类
        format_board = self.get_formate_board()
        # 调用Board类的方法，获得可以做的action
        actions = list(format_board.get_legal_actions(color))
        return actions

    # 根据action更新节点局面
    def update_node_with_action(self,action,color):
        format_board = self.get_formate_board()
        format_board._move(action,color)
        for i in range(8):
            for j in range(8):
                self.board_[i][j] = format_board._board[i][j] 

    # 根据Board类的实例board初始化Node
    def initiate_node_with_board(self,format_board):
        for i in range(8):
            for j in range(8):
                self.board_[i][j] = format_board._board[i][j]

    def is_all_expand(self):
        if len(self.get_valid_actions(self.color)) == len(self.children):
            return True
        else:
            return False
    def print_children(self):
        for node in self.children:
            print(node.visit_times)
    # 用于子节点继承父节点的局面
    def born_from(self,node):
        self.board_ = copy.deepcopy(node.board_)
        self.color = self.oppsite_color(node.color)
        self.round_index = node.round_index + 1


class MCTS(object):
    def __init__(self, board, color, max_iter=100):
        # 这是总棋盘,在一次查询中不会进行更改，使用前必须deepcopy
        self.board = board
        # 这是我方颜色，不会变化
        self.mycolor = color
        self.max_iter = max_iter
    def MCTS_search(self,root_node):
        # 先设置root_node的颜色
        root_node.color = self.mycolor
        # 获取当前时间
        start_time = time.perf_counter()
        # 迭代直到到达时间限制

        # while time.perf_counter() - start_time < timeout:
        for k in range(self.max_iter):
            # 根据TreePolicy对树进行扩展，也就是下一步棋
            expand_node = self.tree_policy(root_node, proportion = (self.max_iter - k) / self.max_iter)
            # 对扩展出的局势进行仿真，计算评分
            reward = self.default_policy(expand_node)
            # 把这个评分返回所有路径上的节点
            self.backup(expand_node, reward)
        # 当时间到，从已经访问过的节点中选一个评分最高的返回对应action
        # best_node = self.best_child(root_node, 1)
        best_node = self.best_child_base(root_node,0)
        return root_node.children[best_node]
    
    # 树的维护算法
    def tree_policy(self, node, proportion=1):
        # 当前节点仍可以下棋时：
        while self.is_terminal(node)==False:
            # 如果当前节点未被探索过
            if not node.is_all_expand():
                # 随机返回一个还没探索过的新节点
                sub_node = self.expand(node)
                return sub_node
            else:
                predict_length = count_remaining(node.board_) * proportion
                if predict_length < 1:
                    predict_length = 1
                if node.color != self.mycolor:
                    node = self.best_child(node, predict_length)
                    # node = self.best_child_base(node, 1 / math.sqrt(2.0))
                # 否则就从子节点中找一个最好，重新开始
                else:
                    node = self.best_child(node,predict_length)
                continue
        return node

    # 随机返回新节点
    def expand(self,node):
        # 根据当前的局势获得可能的action
        action_list = node.get_valid_actions(node.color)
        action = random.choice(action_list)
        
        # 找一个没有探索过的
        while action in node.children.values():
            action = random.choice(action_list)
        # 根据action构建新node
        sub_node = Node()
        sub_node.born_from(node)
        sub_node.update_node_with_action(action,node.color)
        # 把新node添加到原node的children中
        node.children[sub_node]=action
        sub_node.parent = node
        return sub_node
    
    # 根据当前局势进行仿真。使用快速走子策略进行，此处我们使用Random进行快速仿真
    def default_policy(self,node):
        temp_node = copy.deepcopy(node)
        game = Random_Game(temp_node.board_,temp_node.color)
        # 获得游戏结果
        (winner,diff) = game.run()
        # 如果是我方获得胜利，返回1；否则（输或平局）返回0；
        if (winner==0 and self.mycolor=='X') or (winner==1 and self.mycolor=='O'):
            # if self.mycolor == 'X':
            #     reward = count_remaining(temp_node.board_, 'X') - count_remaining(temp_node.board_, 'O')
            # else:
            #     reward = count_remaining(temp_node.board_, 'O') - count_remaining(temp_node.board_, 'X')
            reward = 1
        else:
            # if self.mycolor == 'X':
            #     reward = -(count_remaining(temp_node.board_, 'X') - count_remaining(temp_node.board_, 'O'))
            # else:
            #     reward = -(count_remaining(temp_node.board_, 'O') - count_remaining(temp_node.board_, 'X'))
            reward = 0
        return reward
    
    # 反向传播算法
    def backup(self,node,reward):
        # 从当前节点开始，向上传递
        while node!=None:
            # 将当前节点访问次数+1
            node.visit_times+=1
            # 更新当前节点评分
            node.quality_value+=reward
            node.quality_value_2 = 0.9 * node.quality_value_2 + 0.1 * reward
            # 更新model
            node.model.getReward(node.quality_value_2)
            # 切换到父节点
            node = node.parent
    
    # UCB算法，取置信上限最高的节点。Cp是参数
    # def best_child(self,node,Cp):
    #     best_node = node
    #     max_value = -100000
    #     for sub_node in node.children.keys():
    #         # 如果是轮到我方下棋，那么胜率正常计算
    #         if node.color == self.mycolor:
    #             value = sub_node.quality_value/sub_node.visit_times+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
    #         # 如果是轮到对方下棋，那么胜率要反过来
    #         else:
    #             value = (1-sub_node.quality_value/sub_node.visit_times)+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
    #         if value>max_value:
    #             best_node = sub_node
    #             max_value = value
    #     return best_node
    # StatMax算法，取置信上限最高的节点。Cp是参数
    def roulette(self, node_value):
        total = sum([value for node, value in node_value])
        if total <= 0:
            return random.choice(node_value)[0]
        if total <= 0:
            raise ValueError(f"total {total} <= 0")
        rand = np.random.rand()
        for node, value in node_value:
            if rand < value / total:
                return node
            rand -= value / total
        return node_value[-1][0]
    
    def genetic_ranking(self, node_value):
        node_value.sort(key=lambda x: x[1])
        rank = [np.exp(i+1) for i in range(len(node_value))]
        # print(rank)
        total = sum(rank)
        rand = np.random.rand()
        for i in range(len(node_value)):
            if rand < rank[i] / total:
                return node_value[i][0]
            rand -= rank[i] / total
        return node_value[-1][0]

    def best_child(self, node, predict_length=1):
        best_node = node
        max_value = -np.inf
        rand = np.random.rand()
        # if rand < 0.1:
        #     return random.choice(list(node.children.keys()))
        for sub_node in node.children.keys():
            if sub_node.visit_times < 3:
                return sub_node
        # if node.color != self.mycolor:
        #     predict_length = 1
        # node_value = [(sub_node,sub_node.model.prediction(predict_length)) for sub_node in node.children.keys()]
        # return self.roulette(node_value)
        # return self.genetic_ranking(node_value)
        for sub_node in node.children.keys():
            # print(predict_length)
            # 如果是轮到我方下棋，那么胜率正常计算
            Cp = 1 / math.sqrt(2.0)
            value_base = sub_node.quality_value/sub_node.visit_times+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
            if node.color == self.mycolor:
                value = value_base + sub_node.model.prediction(random.choice(list(range(1, int(predict_length) + 1))))
            # 如果是轮到对方下棋，那么胜率要反过来
            else:
                value = 1 - (value_base + sub_node.model.prediction(random.choice(list(range(1, int(predict_length) + 1)))))
                # Cp = 1/math.sqrt(2.0)
                # value = (1-sub_node.quality_value/sub_node.visit_times)+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
            if value > max_value:
                best_node = sub_node
                max_value = value
        return best_node

    def best_child_base(self,node,Cp):
        best_node = node
        max_value = -100000
        for sub_node in node.children.keys():
            # 如果是轮到我方下棋，那么胜率正常计算
            if node.color == self.mycolor:
                value = sub_node.quality_value/sub_node.visit_times+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
            # 如果是轮到对方下棋，那么胜率要反过来
            else:
                value = (1-sub_node.quality_value/sub_node.visit_times)+Cp*math.sqrt(2*math.log(node.visit_times)/sub_node.visit_times)
            if value>max_value:
                best_node = sub_node
                max_value = value
        return best_node

    # 当本方不能下棋，就不再扩展该树
    def is_terminal(self,node):
        if len(node.get_valid_actions(node.color)) == 0:
            return True
        else:
            return False


        
class AIPlayer:
    def __init__(self, color, max_iter=100):
        self.color = color
        self.max_iter = max_iter

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        
        mcts = MCTS(board,self.color, self.max_iter)
        root_node = Node()
        root_node.initiate_node_with_board(board)
        action = mcts.MCTS_search(root_node)

        # ------------------------------------------------------------------------

        return action
