import numpy as np
import tensorflow as tf
from collections import deque
import random

# 환경 정의
class YahtzeeEnv:
    def __init__(self):
        self.num_dice = 5
        self.num_rolls = 3
        self.scorecard = [None] * 13
        self.dice = [0] * self.num_dice
        self.reset()
    
    def reset(self):
        """게임 상태 초기화"""
        self.num_rolls_left = self.num_rolls
        self.scorecard = [None] * 13
        self.dice = np.random.randint(1, 7, size=self.num_dice)
        return self._get_state()
    
    def _get_state(self):
        """현재 상태 반환"""
        # self.dice는 주사위 값, self.num_rolls_left는 남은 롤 횟수, self.scorecard는 점수 카드
        dice_array = np.array(self.dice, dtype=np.float32)  # 주사위 값 (5,)
        rolls_left_array = np.array([self.num_rolls_left], dtype=np.float32)  # 남은 롤 횟수 (1,)
        scorecard_array = np.array([0 if s is None else s for s in self.scorecard], dtype=np.float32)  # 점수 카드 (13,)
        
        # 세 배열을 연결
        return np.concatenate([dice_array, rolls_left_array, scorecard_array])
    
    def roll(self, reroll_indices):
        """선택된 주사위를 다시 굴림"""
        if self.num_rolls_left <= 0:
            raise ValueError("No rolls left!")
        for i in reroll_indices:
            self.dice[i] = np.random.randint(1, 7)
        self.num_rolls_left -= 1
        return self._get_state()
    
    def score(self, category):
        """점수 기록"""
        if self.scorecard[category] is not None:
            raise ValueError("Category already scored!")
        score = self._calculate_score(category)
        self.scorecard[category] = score
        self.num_rolls_left = self.num_rolls  # Reset rolls for next turn
        self.dice = np.random.randint(1, 7, size=self.num_dice)  # Reset dice
        return score, self._get_state()
    
    def _calculate_score(self, category):
        """카테고리 점수 계산 (간단히 구현)"""
        if category < 6:  # 1~6
            return sum(d for d in self.dice if d == category + 1)
        elif category == 6:  # 3 of a kind
            return sum(self.dice) if any(np.sum(self.dice == d) >= 3 for d in set(self.dice)) else 0
        elif category == 7:  # 4 of a kind
            return sum(self.dice) if any(np.sum(self.dice == d) >= 4 for d in set(self.dice)) else 0
        elif category == 8:  # Full house
            counts = [np.sum(self.dice == d) for d in set(self.dice)]
            return 25 if sorted(counts) == [2, 3] else 0
        elif category == 9:  # Small straight
            unique_values = set(self.dice)
            return 30 if len(unique_values) >= 4 else 0
        elif category == 10:  # Large straight
            unique_values = set(self.dice)
            return 40 if len(unique_values) == 5 and (max(unique_values) - min(unique_values)) == 4 else 0
        elif category == 11:  # Yahtzee
            return 50 if any(np.sum(self.dice == d) == 5 for d in set(self.dice)) else 0
        elif category == 12:  # Chance
            return sum(self.dice)
        return 0

    def is_done(self):
        """게임 종료 확인"""
        return all(s is not None for s in self.scorecard)

# DQL 에이전트 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 상태 크기 (입력층 크기)
        self.action_size = action_size  # 행동 크기 (출력층 크기)
        self.memory = deque(maxlen=2000)  # 경험을 저장하는 메모리
        self.gamma = 0.95  # 할인율 (Future reward의 중요도 조정)
        self.epsilon = 1.0  # 탐험률 (Exploration)
        self.epsilon_min = 0.01  # 탐험률의 최소값
        self.epsilon_decay = 0.995  # 탐험률 감소 비율
        self.learning_rate = 0.001  # 학습률
        self.model = self._build_model()  # Q-Network 생성
    def _build_model(self):
        """Q-Network 구성"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'), #입력층: state_size 크기의 입력을 받습니다.
            tf.keras.layers.Dense(24, activation='relu'), #은닉층: 24개의 뉴런과 ReLU 활성화 함수 사용.
            tf.keras.layers.Dense(self.action_size, activation='linear') # 출력층: action_size 크기의 Q-값을 출력.
        ])
        #손실 함수: mean squared error (MSE)를 사용해 예측 Q-값과 실제 Q-값 간의 차이를 줄입니다.
        #최적화 알고리즘: Adam 최적화 알고리즘을 사용합니다.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """에이전트가 경험한 상태, 행동, 보상, 다음 상태, 종료 여부를 메모리에 저장합니다."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """에이전트는 현재 상태에서 행동을 선택합니다."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # 탐험: epsilon 확률로 랜덤 행동을 선택하여 새로운 상황을 탐색.
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0]) # 활용 : 1-epsilon 확률로 최대 Q-값(출력값)을 가지는 행동을 선택.
    
    def replay(self, batch_size):
        """경험 재학습"""
        minibatch = random.sample(self.memory, batch_size) # 미니배치 샘플링: 메모리에서 무작위로 batch_size개의 경험을 샘플링.
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]) # 현재 상태에서의 Q-값을 업데이트.
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # 모델 업데이트: 업데이트된 Q-값을 사용해 신경망을 업데이트.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # 탐험률 감소: 탐험률을 점차 감소시켜 학습 과정을 안정화.

# 학습 실행
if __name__ == "__main__":
    env = YahtzeeEnv()
    state_size = len(env._get_state())
    action_size = 13  # 13개의 카테고리
    agent = DQNAgent(state_size, action_size)
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        while not env.is_done():
            action = agent.act(state)
            try:
                reward, next_state = env.score(action)
            except ValueError:
                reward, next_state = -10, state  # Invalid action penalty
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            done = env.is_done()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
        
        if len(agent.memory) > 32:
            agent.replay(32)
