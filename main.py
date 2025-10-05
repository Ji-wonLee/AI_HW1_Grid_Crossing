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
    """Grid Crossing을 위한 최적화된 A* + Q-Learning 하이브리드 에이전트"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state(x,y,dir) -> action_values
        self.q_table = {}
        
        # 방문 기록: 순환 방지용
        self.visited_states = {}
        
        # A* 경로 관련
        self.path = None
        self.path_index = 0
        
        # 훈련 모드 플래그
        self.training_mode = True
        
    def act(self, observation: Any, info: Dict) -> Any:
        """환경 관측값을 받아 행동 반환"""
        pos, direction = self._get_player_info(observation)
        if pos is None or direction is None:
            return random.randint(0, 2)
            
        state = (pos[0], pos[1], direction)
        
        # 방문 기록 업데이트
        if self.training_mode:
            self.visited_states[state] = self.visited_states.get(state, 0) + 1
        
        # ε-greedy 정책
        if random.random() < self.epsilon:
            # A* 경로 재계산 조건
            if (self.path is None or 
                self.path_index >= len(self.path) or
                (self.path_index > 0 and pos not in self.path)):
                
                goal = self._get_goal_pos(observation)
                if goal:
                    self.path = self._find_path_astar(pos, goal, observation)
                    self.path_index = 0
            
            # A* 경로 따라가기
            if self.path and self.path_index < len(self.path):
                # 현재 위치가 경로상에 있으면 인덱스 업데이트
                if pos in self.path:
                    self.path_index = self.path.index(pos) + 1
                
                if self.path_index < len(self.path):
                    target_pos = self.path[self.path_index]
                    target_direction = self._get_direction_to_target(pos, target_pos)
                    
                    if target_direction is not None:
                        action = self._get_action_to_direction(direction, target_direction)
                        return action
            
            # A* 실패시 랜덤
            return random.randint(0, 2)
        else:
            # Q-table 활용
            if state in self.q_table:
                return np.argmax(self.q_table[state])
            else:
                return random.randint(0, 2)
    
    def save(self, path: str):
        """에이전트를 파일로 저장"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'min_epsilon': self.min_epsilon
        }
        
        # 디렉토리 생성
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        """저장된 에이전트 로드"""
        agent = cls()
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                agent.q_table = data.get('q_table', {})
                # 평가시 epsilon을 매우 낮게 설정
                agent.epsilon = 0.01  # 거의 greedy하게
                agent.learning_rate = data.get('learning_rate', 0.1)
                agent.discount_factor = data.get('discount_factor', 0.95)
                agent.min_epsilon = data.get('min_epsilon', 0.01)
        
        agent.training_mode = False  # 평가 모드
        return agent
    
    # === 헬퍼 메서드들 ===
    
    def _get_player_info(self, observation):
        """플레이어 위치와 방향 추출"""
        player_coords = np.where((observation >= 1000) & (observation <= 1003))
        
        if len(player_coords[0]) == 0:
            return None, None
            
        x, y = player_coords[0][0], player_coords[1][0]
        direction = observation[x, y] - 1000
        return (x, y), int(direction)
    
    def _get_goal_pos(self, observation):
        """목표 위치 찾기"""
        goal_coords = np.where(observation == 810)
        
        if len(goal_coords[0]) == 0:
            return None
            
        return (goal_coords[0][0], goal_coords[1][0])
    
    def _find_path_astar(self, start, goal, observation):
        """개선된 A* 알고리즘으로 최단 경로 찾기"""
        if start is None or goal is None:
            return []
        
        def heuristic(a, b):
            # 맨해튼 거리
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in closed_set:
                continue
                
            if current == goal:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            closed_set.add(current)
            
            # 이웃 노드 탐색
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 경계 체크
                if not (0 <= neighbor[0] < 26 and 0 <= neighbor[1] < 26):
                    continue
                
                # 지나갈 수 있는 타일인지 확인
                tile_value = observation[neighbor[0], neighbor[1]]
                # 바닥(100), 목표(810), 또는 플레이어가 있던 자리(1000~1003)
                if not (tile_value == 100 or tile_value == 810 or (1000 <= tile_value <= 1003)):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _get_direction_to_target(self, current_pos, target_pos):
        """목표 위치를 향하는 방향 계산 (수정된 방향 매핑)"""
        if current_pos is None or target_pos is None:
            return None
            
        dx = target_pos[1] - current_pos[1]
        dy = target_pos[0] - current_pos[0]
        
        # 방향 매핑: 0=좌, 1=하, 2=우, 3=상
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 0  # 우(2) 또는 좌(0)
        else:
            return 1 if dy > 0 else 3  # 하(1) 또는 상(3)
    
    def _get_action_to_direction(self, current_direction, target_direction):
        """현재 방향에서 목표 방향으로 회전하는 행동 계산"""
        if target_direction is None:
            return random.randint(0, 2)
        
        if current_direction == target_direction:
            return 2  # 전진
        
        # 회전 방향 결정
        diff = (target_direction - current_direction + 4) % 4
        
        if diff == 1:
            return 1  # 오른쪽 회전
        elif diff == 3:
            return 0  # 왼쪽 회전
        else:  # diff == 2 (180도)
            return random.choice([0, 1])  # 둘 중 아무거나
    
    # === 훈련용 메서드들 ===
    
    def get_reward(self, observation, next_observation, done, pos, next_pos):
        """개선된 보상 함수"""
        if done:
            # 성공: 목표가 사라짐
            goal_coords = np.where(next_observation == 810)
            if len(goal_coords[0]) == 0:
                return 100  # 성공 보상
            else:
                return -100  # 실패 (용암 등)
        
        # 방문 패널티 (순환 방지)
        state = (next_pos[0], next_pos[1], next_pos[2]) if next_pos else None
        visit_penalty = 0
        if state and state in self.visited_states:
            visit_penalty = -2 * self.visited_states[state]
        
        # 벽 충돌 패널티
        if pos and next_pos and pos[:2] == next_pos[:2]:
            stuck_penalty = -5
        else:
            stuck_penalty = 0
        
        # 목표까지 거리 변화
        goal = self._get_goal_pos(next_observation)
        if goal and pos and next_pos:
            old_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            new_dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
            distance_reward = (old_dist - new_dist) * 2  # 가까워지면 보상
        else:
            distance_reward = 0
        
        # 시간 패널티
        time_penalty = -0.1
        
        return distance_reward + visit_penalty + stuck_penalty + time_penalty
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Q-learning 업데이트"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        
        old_value = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(3)
            next_max = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * next_max
        
        # Q-value 업데이트
        self.q_table[state][action] = (1 - self.learning_rate) * old_value + self.learning_rate * target
    
    def decay_epsilon(self):
        """엡실론 감소"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train():
    """Grid Crossing 환경에서 에이전트 훈련"""
    print("=" * 60)
    print("Grid Crossing - Optimized A* + Q-Learning Training")
    print("=" * 60)
    
    env = gym.make(
        id='kymnasium/GridAdventure-Crossing-26x26-v0',
        render_mode='rgb_array',
        bgm=False
    )
    
    agent = YourAgent(
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.998,
        min_epsilon=0.01
    )
    
    num_episodes = 5000  # 충분한 학습
    success_count = 0
    recent_successes = deque(maxlen=100)
    
    print(f"\nTraining for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        observation, info = env.reset()
        
        # 에피소드마다 초기화
        agent.visited_states = {}
        agent.path = None
        agent.path_index = 0
        
        total_reward = 0
        
        for step in range(500):
            # 현재 상태
            pos, direction = agent._get_player_info(observation)
            if pos is None:
                break
            state = (pos[0], pos[1], direction)
            
            # 행동 선택
            action = agent.act(observation, info)
            
            # 환경 스텝
            next_observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 다음 상태
            next_pos, next_direction = agent._get_player_info(next_observation)
            if next_pos:
                next_state = (next_pos[0], next_pos[1], next_direction)
            else:
                next_state = None
                done = True
            
            # 보상 계산
            reward = agent.get_reward(
                observation, next_observation, done,
                (pos[0], pos[1], direction) if pos else None,
                (next_pos[0], next_pos[1], next_direction) if next_pos else None
            )
            total_reward += reward
            
            # Q-table 업데이트
            if next_state:
                agent.update_q_table(state, action, reward, next_state, done)
            
            observation = next_observation
            
            # 성공 체크
            if done:
                goal_coords = np.where(next_observation == 810)
                if len(goal_coords[0]) == 0:  # 목표가 사라졌으면 성공
                    success_count += 1
                    recent_successes.append(1)
                    print(f"Episode {episode}: SUCCESS in {step+1} steps! (Total: {success_count})")
                else:
                    recent_successes.append(0)
                break
        
        # 엡실론 감소
        agent.decay_epsilon()
        
        # 주기적 저장 및 상태 출력
        if episode % 100 == 0:
            success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            print(f"\nEpisode {episode}:")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Recent Success Rate: {success_rate:.1%}")
            print(f"  Total Successes: {success_count}")
            
            # 모델 저장
            os.makedirs('models', exist_ok=True)
            agent.save('models/trained_agent.pkl')
            
            # 메인 디렉토리에도 저장 (호환성)
            agent.save('trained_agent.pkl')
    
    # 최종 저장
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Successes: {success_count}/{num_episodes}")
    print(f"Final Success Rate: {success_count/num_episodes:.1%}")
    
    os.makedirs('models', exist_ok=True)
    agent.save('models/trained_agent.pkl')
    agent.save('trained_agent.pkl')
    
    env.close()

if __name__ == "__main__":
    train()