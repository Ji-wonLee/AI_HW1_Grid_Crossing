# ===================================================================
# 파일: evaluate.py
# 용도: 훈련된 에이전트의 성능을 평가하고 시연하는 코드
#       - 성공률, 평균 스텝 수, 실패시 목표까지 거리 측정
#       - 교수님 평가용 demo 기능 포함
# ===================================================================

import gymnasium as gym
import kymnasium as kym
import numpy as np
import sys
from main import YourAgent

def evaluate_agent(agent_path='trained_agent.pkl', num_episodes=10, render=False):
    """에이전트 성능 평가"""
    print("=" * 60)
    print("Grid Crossing Agent Evaluation")
    print("=" * 60)
    
    env = gym.make(
        id='kymnasium/GridAdventure-Crossing-26x26-v0',
        render_mode='human' if render else 'rgb_array',
        bgm=render
    )
    # 에이전트 로드 시도 (models 폴더도 확인)
    try:
        agent = YourAgent.load(agent_path)
        print(f"Agent loaded from: {agent_path}")
    except FileNotFoundError:
        # models 폴더에서도 시도
        try:
            agent = YourAgent.load(f'models/{agent_path}')
            print(f"Agent loaded from: models/{agent_path}")
        except:
            print(f"Error: Agent file not found at {agent_path}")
            return
    
    success_count = 0
    total_steps = []
    failed_distances = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        
        # 평가시마다 경로 초기화
        agent.path = None
        agent.path_index = 0
        
        steps = 0
        is_success = False
        last_player_pos = None
        
        while steps < 500:
            # 플레이어 위치 추적 (실패시 거리 계산용)
            player_coords = np.where((observation >= 1000) & (observation <= 1003))
            if len(player_coords[0]) > 0:
                last_player_pos = (player_coords[0][0], player_coords[1][0])
            
            # 행동 선택 및 실행
            action = agent.act(observation, info)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done:
                # ## 💡 성공 판정: 목표(810)가 맵에서 사라짐 = 도달 성공
                goal_coords = np.where(next_observation == 810)
                if len(goal_coords[0]) == 0:
                    is_success = True
                    success_count += 1
                    total_steps.append(steps)
                    print(f"Episode {episode + 1}: SUCCESS in {steps} steps")
                else:
                    # 실패시 목표까지 거리 계산
                    if last_player_pos and len(goal_coords[0]) > 0:
                        goal_pos = (goal_coords[0][0], goal_coords[1][0])
                        distance = abs(last_player_pos[0] - goal_pos[0]) + abs(last_player_pos[1] - goal_pos[1])
                        failed_distances.append(distance)
                        print(f"Episode {episode + 1}: FAILED (distance to goal: {distance})")
                    else:
                        print(f"Episode {episode + 1}: FAILED")
                break
            
            observation = next_observation
        
        if not done:
            # 500스텝 시간 초과
            goal_coords = np.where(observation == 810)
            if last_player_pos and len(goal_coords[0]) > 0:
                goal_pos = (goal_coords[0][0], goal_coords[1][0])
                distance = abs(last_player_pos[0] - goal_pos[0]) + abs(last_player_pos[1] - goal_pos[1])
                failed_distances.append(distance)
                print(f"Episode {episode + 1}: TIMEOUT (distance to goal: {distance})")
            else:
                print(f"Episode {episode + 1}: TIMEOUT")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    success_rate = (success_count / num_episodes) * 100
    print(f"Success Rate: {success_count}/{num_episodes} ({success_rate:.1f}%)")
    
    if total_steps:
        avg_steps = np.mean(total_steps)
        min_steps = min(total_steps)
        max_steps = max(total_steps)
        print(f"\nSuccessful Episodes:")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Best Performance: {min_steps} steps")
        print(f"  Worst Performance: {max_steps} steps")
        
        # 성능 등급
        if success_rate >= 90 and avg_steps < 100:
            grade = "S (Excellent)"
        elif success_rate >= 80 and avg_steps < 150:
            grade = "A (Great)"
        elif success_rate >= 70 and avg_steps < 200:
            grade = "B (Good)"
        elif success_rate >= 50:
            grade = "C (Fair)"
        else:
            grade = "D (Needs Improvement)"
        
        print(f"\nPerformance Grade: {grade}")
    
    if failed_distances:
        avg_dist = np.mean(failed_distances)
        print(f"\nFailed Episodes:")
        print(f"  Average Distance to Goal: {avg_dist:.1f}")
    
    env.close()

def demo_agent(agent_path='trained_agent.pkl'):
    """시연용 함수 - 교수님 평가 코드용"""
    print("=" * 60)
    print("Grid Crossing Agent Demo")
    print("=" * 60)
    
    try:
        agent = YourAgent.load(agent_path)
        print(f"Agent loaded successfully from: {agent_path}")
    except FileNotFoundError:
        try:
            agent = YourAgent.load(f'models/{agent_path}')
            print(f"Agent loaded successfully from: models/{agent_path}")
        except:
            print(f"Error: Agent file not found")
            return
    
    print("\nStarting demonstration...")
    print("-" * 60)
    
    try:
        kym.evaluate(
            env_id='kymnasium/GridAdventure-Crossing-26x26-v0',
            agent=agent,
            render_mode='human',
            bgm=True
        )
    except Exception as e:
        print(f"Demo error: {e}")

def test_multiple_runs(agent_path='trained_agent.pkl', num_runs=5, episodes_per_run=10):
    """여러 번 테스트하여 평균 성능 확인"""
    print("=" * 60)
    print("Multiple Runs Test")
    print("=" * 60)
    
    all_success_rates = []
    all_avg_steps = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        env = gym.make(
            id='kymnasium/GridAdventure-Crossing-26x26-v0',
            render_mode='rgb_array',
            bgm=False
        )
        
        try:
            agent = YourAgent.load(agent_path)
        except:
            agent = YourAgent.load(f'models/{agent_path}')
        
        success_count = 0
        steps_list = []
        
        for episode in range(episodes_per_run):
            observation, info = env.reset()
            agent.path = None
            steps = 0
            
            while steps < 500:
                action = agent.act(observation, info)
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                
                if terminated or truncated:
                    goal_coords = np.where(observation == 810)
                    if len(goal_coords[0]) == 0:  # Success
                        success_count += 1
                        steps_list.append(steps)
                    break
        
        success_rate = (success_count / episodes_per_run) * 100
        all_success_rates.append(success_rate)
        
        if steps_list:
            avg_steps = np.mean(steps_list)
            all_avg_steps.append(avg_steps)
            print(f"  Success Rate: {success_rate:.1f}%, Avg Steps: {avg_steps:.1f}")
        else:
            print(f"  Success Rate: {success_rate:.1f}%, No successes")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Average Success Rate: {np.mean(all_success_rates):.1f}%")
    if all_avg_steps:
        print(f"Average Steps (when successful): {np.mean(all_avg_steps):.1f}")
    print(f"Success Rate Std Dev: {np.std(all_success_rates):.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'eval':
            agent_path = sys.argv[2] if len(sys.argv) > 2 else 'trained_agent.pkl'
            num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            evaluate_agent(agent_path, num_episodes, render=False)
            
        elif command == 'demo':
            agent_path = sys.argv[2] if len(sys.argv) > 2 else 'trained_agent.pkl'
            demo_agent(agent_path)
            
        elif command == 'test':
            agent_path = sys.argv[2] if len(sys.argv) > 2 else 'trained_agent.pkl'
            test_multiple_runs(agent_path)
            
        else:
            print("Unknown command. Available commands:")
            print("  python evaluate.py eval [agent_path] [num_episodes]")
            print("  python evaluate.py demo [agent_path]")
            print("  python evaluate.py test [agent_path]")
    else:
        print("Grid Crossing Agent Evaluator")
        print("=" * 60)
        print("Usage:")
        print("  python evaluate.py eval     - Evaluate agent performance")
        print("  python evaluate.py demo     - Visual demonstration")
        print("  python evaluate.py test     - Test multiple runs")
        print()
        print("Default: Running evaluation...")
        evaluate_agent()