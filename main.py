# ===================================================================
# Grid Crossing - A* 가이드 Q-Learning 하이브리드 에이전트
# 전략: A*로 최적 경로를 보여주면서 Q-table을 학습시킴
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
    """A* 가이드 기반 Q-Learning 에이전트"""
    
    def __init__(self, learning_rate=0.15, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.01):
        # Q-Learning 하이퍼파라미터
        self.learning_rate = learning_rate      # α: 학습률
        self.discount_factor = discount_factor  # γ: 미래 보상 할인율
        self.epsilon = epsilon                  # ε: 탐험 확률
        self.epsilon_decay = epsilon_decay      # ε 감소율
        self.min_epsilon = min_epsilon          # 최소 ε
        
        # Q-table: (x, y, direction) -> [Q(s,좌회전), Q(s,우회전), Q(s,전진)]
        self.q_table = {}
        
        # 에피소드 내 방문 기록 (순환 방지)
        self.visited_states = {}
        
        # A* 계산 결과 저장
        self.action_sequence = None  # A*가 계산한 최적 행동들
        self.action_index = 0        # 현재 실행 중인 행동 인덱스
        
        # 훈련/평가 모드
        self.training_mode = True
        
    def act(self, observation: Any, info: Dict) -> Any:
        """행동 선택 (ε-greedy)"""
        pos, direction = self._get_player_info(observation)
        if pos is None or direction is None:
            return random.randint(0, 2)
            
        state = (pos[0], pos[1], direction)
        
        # 방문 기록 (훈련 시에만)
        if self.training_mode:
            self.visited_states[state] = self.visited_states.get(state, 0) + 1
        
        # ε-greedy: 확률 ε로 탐험(A*), 확률 1-ε로 활용(Q-table)
        if random.random() < self.epsilon:
            return self._explore_with_astar(observation, pos, direction)
        else:
            return self._exploit_qtable(state)
    
    def _explore_with_astar(self, observation, pos, direction):
        """탐험: A* 알고리즘으로 최적 경로 찾기"""
        # 행동 시퀀스가 없거나 다 썼으면 재계산
        if (self.action_sequence is None or 
            self.action_index >= len(self.action_sequence)):
            
            goal = self._get_goal_pos(observation)
            if goal:
                self.action_sequence = self._find_path_astar(
                    pos, goal, observation, direction
                )
                self.action_index = 0
        
        # A*가 계산한 행동 따라가기
        if self.action_sequence and self.action_index < len(self.action_sequence):
            action = self.action_sequence[self.action_index]
            self.action_index += 1
            return action
        
        # A* 실패 시 랜덤
        return random.randint(0, 2)
    
    def _exploit_qtable(self, state):
        """활용: 학습된 Q-table 사용"""
        if state in self.q_table:
            # Q값이 가장 높은 행동 선택
            return np.argmax(self.q_table[state])
        else:
            # 처음 보는 상태면 랜덤
            return random.randint(0, 2)
    
    def save(self, path: str):
        """모델 저장"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'min_epsilon': self.min_epsilon
        }
        
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        """모델 로드"""
        agent = cls()
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                agent.q_table = data.get('q_table', {})
                agent.epsilon = 0.0  # 평가 시 완전히 greedy
                agent.learning_rate = data.get('learning_rate', 0.15)
                agent.discount_factor = data.get('discount_factor', 0.95)
                agent.min_epsilon = data.get('min_epsilon', 0.01)
        
        agent.training_mode = False
        return agent
    
    # ===============================================================
    # 환경 정보 추출
    # ===============================================================
    
    def _get_player_info(self, observation):
        """플레이어 위치와 방향 추출"""
        # 1000~1003: 플레이어 (1000=우, 1001=하, 1002=좌, 1003=상)
        player_coords = np.where((observation >= 1000) & (observation <= 1003))
        
        if len(player_coords[0]) == 0:
            return None, None
            
        x, y = player_coords[0][0], player_coords[1][0]
        direction = observation[x, y] - 1000  # 0=우, 1=하, 2=좌, 3=상
        return (x, y), int(direction)
    
    def _get_goal_pos(self, observation):
        """목표 위치 찾기"""
        goal_coords = np.where(observation == 810)  # 810: 목표
        
        if len(goal_coords[0]) == 0:
            return None
            
        return (goal_coords[0][0], goal_coords[1][0])
    
    # ===============================================================
    # A* 알고리즘 (최적 경로 찾기)
    # ===============================================================
    
    def _find_path_astar(self, start, goal, observation, start_direction):
        """A* 알고리즘으로 최적 행동 시퀀스 반환"""
        
        def heuristic(pos, goal):
            """휴리스틱: 맨해튼 거리"""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # 우선순위 큐: (f_score, (위치, 방향))
        open_set = [(0, (start, start_direction))]
        
        # 경로 추적용
        came_from = {}       # 어느 상태에서 왔는지
        action_from = {}     # 어떤 행동으로 왔는지
        g_score = {(start, start_direction): 0}  # 시작점부터 거리
        closed_set = set()
        
        while open_set:
            current_f, (current_pos, current_dir) = heapq.heappop(open_set)
            
            # 이미 방문한 상태면 스킵
            if (current_pos, current_dir) in closed_set:
                continue
            
            # 목표 도착!
            if current_pos == goal:
                # 행동 시퀀스 재구성
                actions = []
                state = (current_pos, current_dir)
                while state in came_from:
                    actions.append(action_from[state])
                    state = came_from[state]
                return actions[::-1]  # 뒤집어서 반환
            
            closed_set.add((current_pos, current_dir))
            
            # 가능한 행동 3가지 탐색
            for action in [0, 1, 2]:  # 좌회전, 우회전, 전진
                new_state, valid = self._simulate_action(
                    current_pos, current_dir, action, observation
                )
                
                if not valid:
                    continue
                
                new_pos, new_dir = new_state
                tentative_g = g_score[(current_pos, current_dir)] + 1
                
                # 더 좋은 경로를 찾았으면 업데이트
                if new_state not in g_score or tentative_g < g_score[new_state]:
                    came_from[new_state] = (current_pos, current_dir)
                    action_from[new_state] = action
                    g_score[new_state] = tentative_g
                    f_score = tentative_g + heuristic(new_pos, goal)
                    heapq.heappush(open_set, (f_score, new_state))
        
        return []  # 경로 없음
    
    def _simulate_action(self, pos, direction, action, observation):
        """행동 시뮬레이션"""
        if action == 0:  # 좌회전
            new_dir = (direction - 1) % 4
            return (pos, new_dir), True
            
        elif action == 1:  # 우회전
            new_dir = (direction + 1) % 4
            return (pos, new_dir), True
            
        else:  # 전진 (action == 2)
            # 방향별 이동: 0=우(0,+1), 1=하(+1,0), 2=좌(0,-1), 3=상(-1,0)
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            dx, dy = moves[direction]
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # 경계 체크
            if not (0 <= new_pos[0] < 26 and 0 <= new_pos[1] < 26):
                return None, False
            
            # 장애물 체크 (100=바닥, 810=목표, 1000~1003=플레이어)
            tile = observation[new_pos[0], new_pos[1]]
            if tile == 100 or tile == 810 or (1000 <= tile <= 1003):
                return (new_pos, direction), True
            
            return None, False
    
    # ===============================================================
    # Q-Learning 학습
    # ===============================================================
    
    def get_reward(self, observation, next_observation, done, pos, next_pos):
        """보상 함수"""
        # 종료 상태 보상
        if done:
            goal_coords = np.where(next_observation == 810)
            if len(goal_coords[0]) == 0:  # 목표가 사라짐 = 성공
                return 100
            else:  # 용암에 빠짐 = 실패
                return -100
        
        # 순환 패널티 (같은 곳 계속 방문)
        state = (next_pos[0], next_pos[1], next_pos[2]) if next_pos else None
        visit_penalty = 0
        if state and state in self.visited_states:
            visit_penalty = -2 * self.visited_states[state]
        
        # 벽 충돌 패널티
        stuck_penalty = -5 if (pos and next_pos and pos[:2] == next_pos[:2]) else 0
        
        # 목표 접근 보상
        goal = self._get_goal_pos(next_observation)
        distance_reward = 0
        if goal and pos and next_pos:
            old_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            new_dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
            distance_reward = (old_dist - new_dist) * 2
        
        # 시간 패널티 (빨리 도착하도록)
        time_penalty = -0.1
        
        return distance_reward + visit_penalty + stuck_penalty + time_penalty
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Q-값 업데이트 (Q-Learning 공식)"""
        # Q-table 초기화
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        
        old_q = self.q_table[state][action]
        
        # TD Target 계산
        if done:
            target = reward
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(3)
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Q-값 업데이트: Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        self.q_table[state][action] = old_q + self.learning_rate * (target - old_q)
    
    def decay_epsilon(self):
        """탐험 확률 감소"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ===================================================================
# 훈련 루프
# ===================================================================

def train():
    """에이전트 훈련"""
    print("=" * 60)
    print("Grid Crossing - A* Guided Q-Learning")
    print("=" * 60)
    
    # 환경 생성
    env = gym.make(
        id='kymnasium/GridAdventure-Crossing-26x26-v0',
        render_mode='rgb_array',
        bgm=False
    )
    
    # 에이전트 생성
    agent = YourAgent(
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.998,
        min_epsilon=0.01
    )
    
    num_episodes = 5000
    success_count = 0
    recent_successes = deque(maxlen=100)
    
    print(f"\nTraining for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        observation, info = env.reset()
        
        # 에피소드 초기화
        agent.visited_states = {}
        agent.action_sequence = None
        agent.action_index = 0
        
        for step in range(500):
            # 현재 상태
            pos, direction = agent._get_player_info(observation)
            if pos is None:
                break
            state = (pos[0], pos[1], direction)
            
            # 행동 선택
            action = agent.act(observation, info)
            
            # 환경 실행
            next_observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 다음 상태
            next_pos, next_direction = agent._get_player_info(next_observation)
            next_state = (next_pos[0], next_pos[1], next_direction) if next_pos else None
            
            # 보상 계산
            reward = agent.get_reward(
                observation, next_observation, done,
                (pos[0], pos[1], direction),
                (next_pos[0], next_pos[1], next_direction) if next_pos else None
            )
            
            # Q-table 업데이트
            if next_state:
                agent.update_q_table(state, action, reward, next_state, done)
            
            observation = next_observation
            
            # 성공 여부 체크
            if done:
                goal_coords = np.where(next_observation == 810)
                if len(goal_coords[0]) == 0:  # 성공
                    success_count += 1
                    recent_successes.append(1)
                    print(f"Episode {episode}: SUCCESS in {step+1} steps!")
                else:
                    recent_successes.append(0)
                break
        
        # 탐험 확률 감소
        agent.decay_epsilon()
        
        # 주기적 저장 및 출력
        if episode % 100 == 0:
            success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            print(f"\nEpisode {episode}:")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Recent 100 Success Rate: {success_rate:.1%}")
            print(f"  Total Successes: {success_count}")
            
            os.makedirs('models', exist_ok=True)
            agent.save('models/trained_agent.pkl')
            agent.save('trained_agent.pkl')
    
    # 최종 결과
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Successes: {success_count}/{num_episodes}")
    print(f"Final Success Rate: {success_count/num_episodes:.1%}")
    
    env.close()


if __name__ == "__main__":
    train()