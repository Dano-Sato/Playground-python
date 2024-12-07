from enum import Enum, auto
import numpy as np
from sklearn.datasets import make_circles, make_moons

class DatasetType(Enum):
    """데이터셋 종류를 정의하는 Enum 클래스"""
    AND = auto()
    OR = auto()
    XOR = auto()
    NAND = auto()
    CIRCLE = auto()
    MOON = auto()
    SINE = auto()
    
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
            DatasetType.SINE: self._create_sine_data()
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
        """사인 함수 데이터"""
        X = np.linspace(0, 2*np.pi, n_samples).reshape(-1, 1)
        y = np.sin(X)
        return X, y
    
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

# 사용 예시
if __name__ == "__main__":
    dm = DatasetManager()
    
    # Enum을 직접 사용
    X, y = dm.get_dataset(DatasetType.OR)    
    nn = DeepNeuralNetwork([2, 4, 4, 1], learning_rate=0.1)
    nn.train(X, y)

    # 예측 결과 출력
    predictions = nn.predict(X)
    for input_val, pred in zip(X, predictions):
        print(f"입력: {input_val}, 예측값: {pred[0]:.4f}")
    

    # 사용 가능한 데이터셋 확인
    print("사용 가능한 데이터셋:", dm.list_datasets())
