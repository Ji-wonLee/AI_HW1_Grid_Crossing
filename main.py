# ===================================================================
# 파일: main.py
# 최종 전략: A* 길찾기를 메인 전략으로 사용하고, Q-러닝을 보조로 활용하며,
#            '상태 방문 기록'을 통해 루프 현상을 방지하는 하이브리드 모델
# ===================================================================

import gymnasium as gym
import kymnasium as kym
import numpy as np
import pickle
import random
from collections import deque
from typing import Any, Dict
import heapq
import os

class YourAgent(kym.Agent):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: (x, y, direction) -> action_values
        self.q_table = {}
        # 방문 기록: (x, y, direction) -> count
        self.visited_states = {}
        
        # A* 경로 관련 변수
        self.path = None
        self.path_index = 0
        
    def act(self, observation: Any, info: Dict) -> Any:
        pos, direction = self._get_player_info(observation)
        state = (pos, direction)
        
        # 현재 상태를 방문 기록에 추가
        self.visited_states[state] = self.visited_states.get(state, 0) + 1

        # ε-greedy 정책
        if random.random() < self.epsilon:
            # A* 경로가 없거나, 끝났거나, 벗어났으면 새로운 경로 탐색
            if self.path is None or self.path_index >= len(self.path):
                goal = self._get_goal_pos(observation)
                if pos and goal:
                    self.path = self._find_path_astar(pos, goal, observation)
                    self.path_index = 0
            
            # 경로가 있으면 따라가기
            if self.path and self.path_index < len(self.path):
                target_pos = self.path[self.path_index]
                if pos == target_pos:
                    self.path_index += 1
                
                if self.path_index < len(self.path):
                    next_target_pos = self.path[self.path_index]
                    target_direction = self._get_direction_to_target(pos, next_target_pos)
                    action = self._get_action_to_direction(direction, target_direction)
                    return action

            # 경로를 못찾거나 문제가 생기면 무작위 행동
            return random.randint(0, 2)
        else:
            # Q-table 활용 (보조)
            q_values = self.q_table.get(state, np.zeros(3))
            return np.argmax(q_values)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
    
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        agent = cls()
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                agent.q_table = data['q_table']
                agent.epsilon = data.get('epsilon', agent.min_epsilon)
        agent.training_mode = False
        return agent

    # === 헬퍼 메서드들 ===
    def _get_player_info(self, obs):
        coords = np.where((obs >= 1000) & (obs <= 1003))
        if len(coords[0]) == 0: return None, None
        pos = (coords[0][0], coords[1][0])
        direction = obs[pos] - 1000
        return pos, int(direction)

    def _get_goal_pos(self, obs):
        coords = np.where(obs == 810)
        if len(coords[0]) == 0: return None
        return (coords[0][0], coords[1][0])
    
    def _find_path_astar(self, start, goal, observation):
        # (이전과 동일한 A* 알고리즘)
        if start is None or goal is None: return []
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        open_set = [(heuristic(start, goal), start)]
        came_from = {}
        g_score = {start: 0}
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < 26 and 0 <= neighbor[1] < 26 and observation[neighbor] in [100, 810]:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        return []

    def _get_direction_to_target(self, current_pos, target_pos):
        dx = target_pos[1] - current_pos[1]; dy = target_pos[0] - current_pos[0]
        if abs(dx) > abs(dy): return 0 if dx > 0 else 2 # 0:우, 2:좌
        else: return 1 if dy > 0 else 3 # 1:하, 3:상

    def _get_action_to_direction(self, current_dir, target_dir):
        if current_dir == target_dir: return 2
        diff = (target_dir - current_dir + 4) % 4
        return 1 if diff == 1 else 0

    # === 훈련용 메서드들 ===
    def get_reward(self, obs, next_obs, done, state, next_state):
        if done:
            return 100 if 810 not in next_obs else -100
        
        # 방문 횟수에 따른 벌점
        visit_penalty = - (self.visited_states.get(next_state, 0) - 1) * 2
        
        # 벽 충돌 벌점
        pos, _ = self._get_player_info(obs)
        next_pos, _ = self._get_player_info(next_obs)
        stuck_penalty = -5 if pos == next_pos else 0
        
        return visit_penalty + stuck_penalty - 0.1 # 시간 벌점

    def update_q_table(self, state, action, reward, next_state, done):
        old_value = self.q_table.get(state, np.zeros(3))[action]
        next_max = np.max(self.q_table.get(next_state, np.zeros(3)))
        
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max * (1-done))
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train():
    env = gym.make(id='kymnasium/GridAdventure-Crossing-26x26-v0', render_mode='rgb_array')
    agent = YourAgent()
    num_episodes = 3000
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        agent.visited_states = {} # 에피소드마다 방문 기록 초기화
        agent.path = None # 경로 초기화
        
        for step in range(500):
            pos, direction = agent._get_player_info(obs)
            if pos is None: break
            state = (pos, direction)
            
            action = agent.act(obs, info)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_pos, next_dir = agent._get_player_info(next_obs)
            if next_pos is None: next_state = None; done = True
            else: next_state = (next_pos, next_dir)
                
            reward = agent.get_reward(obs, next_obs, done, state, next_state)
            
            agent.update_q_table(state, action, reward, next_state, done)
            
            obs = next_obs
            if done:
                if 810 not in next_obs: print(f"Episode {episode}: Success in {step+1} steps!")
                break
        
        agent.decay_epsilon()
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.3f}")
            agent.save('trained_agent.pkl')

if __name__ == "__main__":
    train()

