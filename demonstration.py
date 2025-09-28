import kymnasium as kym
# main.py 파일에 있는 YourAgent 클래스를 가져옵니다.
from main import YourAgent

# 훈련을 통해 저장된 최종 모델 파일을 불러옵니다.
print("🚀 최종 모델을 불러옵니다...")
agent = YourAgent.load('models/grid_crossing_agent_final.pth')
print("✅ 모델 로딩 완료!")

# 공식 시연 함수를 실행합니다.
print("🎬 시연을 시작합니다...")
kym.evaluate(
    env_id='kymnasium/GridAdventure-Crossing-26x26-v0',
    agent=agent,
    render_mode='human',
    bgm=True
)