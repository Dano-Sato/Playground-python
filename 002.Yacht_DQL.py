'''
FUCKED UP
(D)DQN 에이전트를 사용하여 야치 게임을 학습하는 코드입니다.
실패했습니다. 단순히 하면 안되는 행동을 가르치는 것조차 실패했습니다.
(D)DQN은 인간의 보상 설계에 대한 의존성이 매우 높습니다. 거기에 학습 기간도 너무 길어요.
많은 도메인 지식이 필요하고, 사람이 많이 개입해야 합니다. 간단한 문제조차 많은 시행착오가 필요합니다.
이 예제는 버리고 새로운 최적화 방식을 찾아보겠습니다.
'''


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

    def step(self, action):
        """액션 실행 및 다음 상태, 보상 반환"""
        if action < 32:  # 주사위 재굴림 액션 (0~31)
            if self.num_rolls_left <= 0:
                return -1, self._get_state()  # 페널티 완화 (-10 -> -1)
            
            # 재굴림 전 최대 잠재 점수 계산
            old_potential_scores = [self._calculate_score(i) for i in range(13) 
                                  if self.scorecard[i] is None]
            old_max_potential = max(old_potential_scores) if old_potential_scores else 0
            
            # 주사위 재굴림
            reroll_mask = [int(x) for x in format(action, '05b')]
            reroll_indices = [i for i, mask in enumerate(reroll_mask) if mask]
            next_state = self.roll(reroll_indices)
            
            # 재굴림 후 최대 잠재 점수 계산
            new_potential_scores = [self._calculate_score(i) for i in range(13) 
                                  if self.scorecard[i] is None]
            new_max_potential = max(new_potential_scores) if new_potential_scores else 0
            
            # 점수 개선도에 따른 보상
            improvement = new_max_potential - old_max_potential
            reward = improvement * 0.1  # 스케일링
            
            return reward, next_state
        
        else:  # 점수 기록 액션 (32~44)
            category = action - 32
            try:
                score, next_state = self.score(category)
                # 높은 점수에 대해 더 큰 보상
                normalized_reward = score / 50.0  # Yahtzee 최대 점수로 정규화
                return normalized_reward, next_state
            except ValueError:
                return -1, self._get_state()  # 페널티 완화 (-10 -> -1)

# DQL 에이전트 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 야치 게임의 복잡도를 고려한 적절한 메모리 크기
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.learning_rate = 0.005
        self.batch_size = 32  # 안정적인 학습을 위한 적절한 배치 크기
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    def _build_model(self):
        """Q-Network 구성"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        #손실 함수: mean squared error (MSE)를 사용해 예측 Q-값과 실제 Q-값 간의 차이를 줄입니다.
        #최적화 알고리즘: Adam 최적화 ��고리즘을 사용합니다.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=1.0), loss='huber')
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
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        """경험 재학습 (Double DQN)"""
        # 1. Replay Buffer에서 미니배치 샘플링
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # 2. 타겟 값 계산
            target = reward
            if not done:
                # 메인 네트워크로 최적 행동 선택
                best_action = np.argmax(self.model.predict(next_state, verbose=0)[0])

                # 타겟 네트워크로 해당 행동의 Q값 예측
                target_q_value = self.target_model.predict(next_state, verbose=0)[0][best_action]

                # 타겟 업데이트
                target += self.gamma * target_q_value

            # 3. Q값 업데이트
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            # 4. 모델 학습
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # 5. 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 학습 실행
if __name__ == "__main__":
    env = YahtzeeEnv()
    state_size = len(env._get_state())
    action_size = 45  # 32(재굴림 조합) + 13(점수 카테고리)
    agent = DQNAgent(state_size, action_size)
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        while not env.is_done():
            action = agent.act(state)
            reward, next_state = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            done = env.is_done()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
        
        if len(agent.memory) > 32:
            agent.replay(32)
