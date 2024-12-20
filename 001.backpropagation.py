from enum import Enum, auto
import numpy as np
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt

class DatasetType(Enum):
    """데이터셋 종류를 정의하는 Enum 클래스"""
    AND = auto()
    OR = auto()
    XOR = auto()
    NAND = auto()
    CIRCLE = auto()
    MOON = auto()
    SINE = auto()
    SPIRAL = auto()    # 나선형 데이터
    GAUSSIAN = auto()  # 가우시안 분포 데이터
    DONUT = auto()     # 도넛 모양 데이터
    
    @classmethod
    def from_string(cls, s: str):
        """문자열로부터 Enum 값을 반환"""
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown dataset type: {s}")

class DatasetManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """데이터셋 초기화"""
        self.datasets = {
            DatasetType.AND: self._create_and_gate(),
            DatasetType.OR: self._create_or_gate(),
            DatasetType.XOR: self._create_xor_gate(),
            DatasetType.NAND: self._create_nand_gate(),
            DatasetType.CIRCLE: self._create_circle_data(),
            DatasetType.MOON: self._create_moon_data(),
            DatasetType.SINE: self._create_sine_data(),
            DatasetType.SPIRAL: self._create_spiral_data(),
            DatasetType.GAUSSIAN: self._create_gaussian_data(),
            DatasetType.DONUT: self._create_donut_data()
        }
    
    def _create_and_gate(self):
        """AND 게이트 데이터"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        return X, y
    
    def _create_or_gate(self):
        """OR 게이트 데이터"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        return X, y
    
    def _create_xor_gate(self):
        """XOR 게이트 데이터"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X, y
    
    def _create_nand_gate(self):
        """NAND 게이트 데이터"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [1], [1], [0]])
        return X, y
    
    def _create_circle_data(self, n_samples=1000):
        """원형 데이터"""
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5)
        return X, y.reshape(-1, 1)
    
    def _create_moon_data(self, n_samples=1000):
        """초승달 형태 데이터"""
        X, y = make_moons(n_samples=n_samples, noise=0.1)
        return X, y.reshape(-1, 1)
        
    def _create_sine_data(self, n_samples=1000):
        """사인 함수 데이터 - 출력값을 0~1 범위로 정규화"""
        X = np.linspace(0, 2*np.pi, n_samples).reshape(-1, 1)
        y = np.sin(X)
        # -1~1 범위를 0~1 범위로 변환
        y = (y + 1) / 2
        return X, y
    
    def _create_spiral_data(self, n_samples=1000):
        """나선형 데이터 생성"""
        n = n_samples // 2
        
        # 첫 번째 나선
        theta = np.linspace(0, 4*np.pi, n)
        r = theta + 1
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        
        # 두 번째 나선
        theta = theta + np.pi
        x2 = r * np.cos(theta)
        y2 = r * np.sin(theta)
        
        # 노이즈 추가
        noise = 0.2
        x1 += np.random.randn(n) * noise
        y1 += np.random.randn(n) * noise
        x2 += np.random.randn(n) * noise
        y2 += np.random.randn(n) * noise
        
        # 데이터 합치기
        X = np.vstack([np.column_stack((x1, y1)), np.column_stack((x2, y2))])
        y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
        
        # 데이터 섞기
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]
    
    def _create_gaussian_data(self, n_samples=1000):
        """가우시안 분포 데이터 생성"""
        n = n_samples // 2
        
        # 첫 번째 클래스: 3개의 가우시안
        centers = [(0, 0), (2, 2), (-2, 2)]
        X1 = np.vstack([np.random.randn(n//3, 2) * 0.5 + center 
                       for center in centers])
        y1 = np.zeros(n)
        
        # 두 번째 클래스: 2개의 가우시안
        centers = [(0, 2), (0, -2)]
        X2 = np.vstack([np.random.randn(n//2, 2) * 0.5 + center 
                       for center in centers])
        y2 = np.ones(n)
        
        # 데이터 합치기
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2]).reshape(-1, 1)
        
        # 데이터 섞기
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]
    
    def _create_donut_data(self, n_samples=1000):
        """도넛 모양 데이터 생성"""
        n = n_samples // 2
        
        # 내부 원
        theta = np.random.uniform(0, 2*np.pi, n)
        r = np.random.normal(1, 0.1, n)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        
        # 외부 원
        theta = np.random.uniform(0, 2*np.pi, n)
        r = np.random.normal(3, 0.1, n)
        x2 = r * np.cos(theta)
        y2 = r * np.sin(theta)
        
        # 데이터 합치기
        X = np.vstack([np.column_stack((x1, y1)), np.column_stack((x2, y2))])
        y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
        
        # 데이터 섞기
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]
    
    def get_dataset(self, dataset_type, n_samples=1000):
        """데이터셋 반환"""
        if isinstance(dataset_type, str):
            dataset_type = DatasetType.from_string(dataset_type)
            
        if dataset_type not in DatasetType:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        if dataset_type in [DatasetType.CIRCLE, DatasetType.MOON, DatasetType.SINE]:
            return self._create_dataset(dataset_type, n_samples)
        return self.datasets[dataset_type]
    
    def _create_dataset(self, dataset_type, n_samples):
        """동적 데이터셋 생성"""
        if dataset_type == DatasetType.CIRCLE:
            return self._create_circle_data(n_samples)
        elif dataset_type == DatasetType.MOON:
            return self._create_moon_data(n_samples)
        elif dataset_type == DatasetType.SINE:
            return self._create_sine_data(n_samples)
    
    def list_datasets(self):
        """사용 가능한 데이터셋 목록 반환"""
        return [dtype.name for dtype in DatasetType]

class DeepNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        layer_sizes: 각 층의 뉴런 수를 담은 리스트
        예: [2, 3, 4, 1] -> 입력층(2) -> 은닉층1(3) -> 은닉층2(4) -> 출력층(1)
        """
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Adam 최적화를 위한 하이퍼파라미터
        self.beta1 = 0.9  # 1차 모멘텀 계수
        self.beta2 = 0.999  # 2차 모멘텀 계수
        self.epsilon = 1e-8
        
        # 가중치와 편향 초기화 (He 초기화)
        self.weights = {}
        self.biases = {}
        self.m_weights = {}  # 1차 모멘텀
        self.v_weights = {}  # 2차 모멘텀
        self.m_biases = {}
        self.v_biases = {}
        
        for i in range(self.num_layers - 1):
            scale = np.sqrt(2. / layer_sizes[i])
            self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            self.biases[i] = np.zeros((1, layer_sizes[i+1]))
            
            # Adam 최적화를 위한 모멘텀 초기화
            self.m_weights[i] = np.zeros_like(self.weights[i])
            self.v_weights[i] = np.zeros_like(self.weights[i])
            self.m_biases[i] = np.zeros_like(self.biases[i])
            self.v_biases[i] = np.zeros_like(self.biases[i])
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward_propagation(self, X):
        """순전파"""
        self.activations = {0: X}  # 각 층의 활성화값 저장
        self.z_values = {}         # 각 층의 가중합 저장
        
        activation = X
        for i in range(self.num_layers - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values[i] = z
            activation = self.sigmoid(z)
            self.activations[i+1] = activation
            
        return activation
    
    def backward_propagation(self, X, y):
        """역전파"""
        m = X.shape[0]
        deltas = {}
        
        # 출력층 델타 계산
        deltas[self.num_layers-2] = (self.activations[self.num_layers-1] - y) * \
                                   self.sigmoid_derivative(self.activations[self.num_layers-1])
        
        # 은닉층들의 델타 계산
        for i in range(self.num_layers-3, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * \
                       self.sigmoid_derivative(self.activations[i+1])
        
        # 가중치와 편향 업데이트
        for i in range(self.num_layers-1):
            self.weights[i] -= self.learning_rate * \
                             np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * \
                             np.sum(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs=10000):
        """Adam 최적화를 사용한 신경망 학습"""
        t = 0  # 타임스텝
        
        for epoch in range(epochs):
            t += 1  # 타임스텝 증가
            
            # 순전파
            output = self.forward_propagation(X)
            
            # 손실 계산
            loss = np.mean((y - output) ** 2)
            
            # 역전파
            deltas = {}
            
            # 출력층 델타 계산
            deltas[self.num_layers-2] = (self.activations[self.num_layers-1] - y) * \
                                       self.sigmoid_derivative(self.activations[self.num_layers-1])
            
            # 은닉층들의 델타 계산
            for i in range(self.num_layers-3, -1, -1):
                deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * \
                           self.sigmoid_derivative(self.activations[i+1])
            
            # Adam 최적화를 사용한 가중치와 편향 업데이트
            for i in range(self.num_layers-1):
                # 가중치에 대한 그래디언트
                grad_w = np.dot(self.activations[i].T, deltas[i])
                grad_b = np.sum(deltas[i], axis=0, keepdims=True)
                
                # 가중치의 모멘텀 업데이트
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grad_w
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grad_w ** 2)
                
                # 편향의 모멘텀 업데이트
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grad_b
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grad_b ** 2)
                
                # 편향 보정
                m_hat_w = self.m_weights[i] / (1 - self.beta1 ** t)
                v_hat_w = self.v_weights[i] / (1 - self.beta2 ** t)
                m_hat_b = self.m_biases[i] / (1 - self.beta1 ** t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2 ** t)
                
                # 가중치와 편향 업데이트
                self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            
            # 학습 진행상황 출력
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """예측"""
        return self.forward_propagation(X)

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """결정 경계 시각화 함수"""
    # 그리드 생성
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # 그리드 포인트에 대한 예측
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 시각화
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Prediction Probability')
    plt.show()
# 사용 예시
if __name__ == "__main__":
    dm = DatasetManager()
    test_dataset = DatasetType.GAUSSIAN
    
    # Enum을 직접 사용
    X, y = dm.get_dataset(test_dataset)    
    nn = DeepNeuralNetwork([X.shape[1], 32,16, 1], learning_rate=0.1)
    nn.train(X, y)

   # 결과 시각화
    predictions = nn.predict(X)

    if X.shape[1] == 1:  # 1차원 입력인 경우        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, c='b', label='Input data', alpha=0.5)
        plt.scatter(X, predictions, c='r', label='Predictions', alpha=0.5)
        plt.legend()
        plt.title(f'{test_dataset.name} Function Prediction')
        plt.show()
    elif X.shape[1] == 2:  # 2차원 입력인 경우
        # 결정 경계 시각화
        plot_decision_boundary(nn, X, y, 
                             title=f'{test_dataset.name} - Decision Boundary')

    # 사용 가능한 데이터셋 확인
    print("사용 가능한 데이터셋:", dm.list_datasets())
