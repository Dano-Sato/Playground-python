import numpy as np

# 데이터 초기화
np.random.seed(42)  # 재현성 확보
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력 (AND 문제 예제)
y = np.array([[0], [0], [0], [1]])  # 출력 (AND의 결과)

# 하이퍼파라미터
learning_rate = 0.1
epochs = 10000

# 가중치 초기화 (입력층 → 은닉층, 은닉층 → 출력층)
W1 = np.random.randn(2, 2)  # 입력 → 은닉층 가중치 (2x2)
b1 = np.zeros((1, 2))       # 은닉층 편향
W2 = np.random.randn(2, 1)  # 은닉층 → 출력층 가중치 (2x1)
b2 = np.zeros((1, 1))       # 출력층 편향

# 활성화 함수 (시그모이드 함수)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 시그모이드의 미분
def sigmoid_derivative(z):
    return z * (1 - z)

# 학습 과정
for epoch in range(epochs):
    # 1. 순전파 (Forward Propagation)
    z1 = np.dot(x, W1) + b1         # 입력층 → 은닉층 합
    a1 = sigmoid(z1)                # 은닉층 활성화
    z2 = np.dot(a1, W2) + b2        # 은닉층 → 출력층 합
    a2 = sigmoid(z2)                # 출력층 활성화 (최종 출력)

    # 2. 손실 함수 (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)

    # 3. 역전파 (Backpropagation)
    # 출력층 → 은닉층 (오차의 기울기)
    dL_da2 = a2 - y
    da2_dz2 = sigmoid_derivative(a2)
    dz2_dW2 = a1
    dz2_db2 = 1
    delta2 = dL_da2 * da2_dz2

    # 은닉층 → 입력층 (오차의 기울기)
    dL_da1 = np.dot(delta2, W2.T)
    da1_dz1 = sigmoid_derivative(a1)
    dz1_dW1 = x
    delta1 = dL_da1 * da1_dz1

    # 4. 가중치 및 편향 업데이트
    W2 -= learning_rate * np.dot(dz2_dW2.T, delta2)
    b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(dz1_dW1.T, delta1)
    b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True)

    # 학습 진행 상황 출력
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 최종 결과 출력
print("\nTraining Complete!")
print("Final Loss:", loss)
print("Predicted Outputs:")
print(a2)
