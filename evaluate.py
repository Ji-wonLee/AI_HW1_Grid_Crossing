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
    
    try:
        agent = YourAgent.load(agent_path)
        print(f"Agent loaded from: {agent_path}")
    except FileNotFoundError:
        print(f"Error: Agent file not found at {agent_path}")
        return
    
    success_count = 0
    total_steps = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        agent.path = None # 평가 시작 시 경로 초기화
        steps = 0
        is_success = False
        
        while steps < 500:
            action = agent.act(observation, info)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            observation = next_observation

            # ## 💡 [버그 수정] 성공 판정 로직 개선
            # 게임이 끝났고(done), 그 이유가 성공(810이 맵에 없음)일 때
            if done:
                if 810 not in next_observation:
                    is_success = True
                break # 게임이 끝나면 루프 탈출

        if is_success:
            success_count += 1
            total_steps.append(steps)
            print(f"Episode {episode + 1}: SUCCESS in {steps} steps")
        else:
            print(f"Episode {episode + 1}: FAILED")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    success_rate = (success_count / num_episodes) * 100
    print(f"Success Rate: {success_count}/{num_episodes} ({success_rate:.1f}%)")
    
    if total_steps:
        avg_steps = np.mean(total_steps)
        print(f"Average Steps (on success): {avg_steps:.1f}")
    
    env.close()

def demo_agent(agent_path='trained_agent.pkl'):
    """시연용 함수"""
    print("=" * 60)
    print("Grid Crossing Agent Demo")
    print("=" * 60)
    
    try:
        agent = YourAgent.load(agent_path)
        print(f"Agent loaded successfully from: {agent_path}")
    except FileNotFoundError:
        print(f"Error: Agent file not found at {agent_path}")
        return
    
    print("\nStarting demonstration...")
    kym.evaluate(
        env_id='kymnasium/GridAdventure-Crossing-26x26-v0',
        agent=agent,
        render_mode='human',
        bgm=True
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'eval':
            evaluate_agent()
        elif command == 'demo':
            demo_agent()
    else:
        evaluate_agent()

