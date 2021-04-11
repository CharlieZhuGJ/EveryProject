from graphviz import Digraph
import queue
import random
import math
import hashlib
import logging
import argparse

"""
A quick Monte Carlo Tree Search implementation.  
For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  
The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  
Some features of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR = 1 / math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class State(object):
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=None, turn=NUM_TURNS):
        if moves is None:
            moves = []
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        nextmove = random.choice([x * self.turn for x in self.MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove], self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):
        r = 1.0 - abs((self.value - self.GOAL) / self.MAX_VALUE)  # 越是靠近0，奖励越大
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


class Node(object):
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s


def uct_search(budget, root):
    """
      实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，
      然后返回只要exploitation最高的子节点。
      蒙特卡洛树搜索包含四个步骤：Selection、Expansion、Simulation、Backpropagation。
      前两步使用tree policy找到值得探索的节点。
      第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
      最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
      进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。

    :param budget: 
    :param root: 
    :return: 
    """
    for iteration in range(int(budget)):
        if iteration % 10000 == 9999:
            logger.info("simulation: %d" % iteration)
            logger.info(root)
        front = tree_policy(root)
        reward = default_policy(front.state)
        backup(front, reward)
    return best_child(root, 0)


def tree_policy(node):
    """
    a hack to force 'exploitation' in a game where there are many options,
    and you may never/not want to fully expand first
    蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），
    根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
    基本策略是先找当前未选择过的子节点，如果有多个则随机选。
    如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
    :param node:
    :return:
    """
    while not node.state.terminal():  # 树未搜索到叶节点
        if len(node.children) == 0:
            return expand(node)
        elif random.uniform(0, 1) < .5:
            node = best_child(node, SCALAR)
        else:
            if not node.fully_expanded():
                return expand(node)
            else:
                node = best_child(node, SCALAR)
    return node


def expand(node):
    """
    对某个节点进行扩展
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。
    注意，需要保证新增的节点与其他节点Action不同。
    :param node:
    :return:
    """
    tried_children = [chid.state for chid in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


def best_child(node, scalar):
    """
    选择最有价值节点的策略
    使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
    current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)

    图中N表示总模拟次数，W表示胜局次数。每次都选择胜率最大的节点进行模拟。但是这样会导致新节点无法被探索到。
    为了在最大胜率和新节点探索上保持平衡，UCT（Upper Confidence Bound，上限置信区间算法）被引入

    w/n + scaler(lnN/n)

    :param node:
    :param scalar:
    :return:
    """
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        # 计算UCB
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)


def default_policy(state):
    """
    获得在某种情况下的收益，并更新整颗搜索树的状态
    蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，
    返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
  基本策略是随机选择Action。
    :param state:
    :return:
    """
    while not state.terminal():
        state = state.next_state()
    return state.reward()


def backup(node, reward):
    """
    蒙特卡洛树搜索的Backpropagation阶段，
    输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
    :param node:
    :param reward:
    :return:
    """
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


def show_search_tree(root):
    """
   最后将模型存储在a.dot文件中，将网络结构渲染成 search_path.pdf 文件中。
    :param root:
    :return:
    """
    dot = Digraph(comment='Game Search Tree')
    # if isinstance(root, Node):
    #     dot.node(root.state.node_id(), root.node_info())
    #
    #     que = queue.Queue()
    #     que.put(root.children[0])
    #
    #     while not que.empty():
    #         child = que.get()
    #         if isinstance(child, Node):
    #             dot.node(child.state.node_id(), child.node_info())
    #             dot.edge(child.parent.state.node_id(), child.state.node_id())
    #             for c in child.children:
    #                 que.put(c)
    # with open("a.dot", "w", encoding="utf-8") as writer:
    #     writer.write(dot.source)
    # dot.render('search_path', view=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=False, type=int)
    parser.add_argument('--levels', action="store", required=False, type=int, choices=range(State.NUM_TURNS))
    args = parser.parse_args()
    args.levels = 8
    args.num_sims = 500

    current_node = Node(State())
    for level in range(args.levels):
        current_node = uct_search(args.num_sims / (level + 1), current_node)
        print("level %d" % level)
        print("Num Children: %d" % len(current_node.children))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)

        print("--------------------------------")
    root = current_node
    while root.parent is not None:
        root = root.parent
    print(root)
    print(current_node)
    show_search_tree(root)


