import numpy as np
from sklearn.datasets import make_circles, make_moons

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
            'AND': self._create_and_gate(),
            'OR': self._create_or_gate(),
            'XOR': self._create_xor_gate(),
            'NAND': self._create_nand_gate(),
            'CIRCLE': self._create_circle_data(),
            'MOON': self._create_moon_data(),
            'SINE': self._create_sine_data()
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
    
    def get_dataset(self, name, n_samples=1000):
        """데이터셋 반환"""
        if name.upper() not in self.datasets:
            raise ValueError(f"Unknown dataset: {name}")
            
        if name.upper() in ['CIRCLE', 'MOON', 'SINE']:
            return self._create_dataset(name.upper(), n_samples)
        return self.datasets[name.upper()]
    
    def _create_dataset(self, name, n_samples):
        """동적 데이터셋 생성"""
        if name == 'CIRCLE':
            return self._create_circle_data(n_samples)
        elif name == 'MOON':
            return self._create_moon_data(n_samples)
        elif name == 'SINE':
            return self._create_sine_data(n_samples)
    
    def list_datasets(self):
        """사용 가능한 데이터셋 목록 반환"""
        return list(self.datasets.keys())

class DeepNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        layer_sizes: 각 층의 뉴런 수를 담은 리스트
        예: [2, 3, 4, 1] -> 입력층(2) -> 은닉층1(3) -> 은닉층2(4) -> 출력층(1)
        """
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # 가중치와 편향 초기화
        self.weights = {}
        self.biases = {}

        # Xavier 초기화        
        for i in range(self.num_layers - 1):
            scale = np.sqrt(2. / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            self.biases[i] = np.zeros((1, layer_sizes[i+1]))
    
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
    
    def train(self, X, Y, epochs=10000):
        """신경망 학습"""
        for epoch in range(epochs):
            # 순전파
            output = self.forward_propagation(X)
            
            # 손실 계산
            loss = np.mean((Y - output) ** 2)
            
            # 역전파
            self.backward_propagation(X, Y)
            
            # 학습 진행상황 출력
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """예측"""
        return self.forward_propagation(X)

# 사용 예시
if __name__ == "__main__":
    # 데이터셋 매니저 인스턴스 생성
    dm = DatasetManager()
    
    # XOR 게이트 학습 예제
    X, y = dm.get_dataset('XOR')
    nn = DeepNeuralNetwork([2, 4, 4, 1], learning_rate=0.1)
    nn.train(X, y)

    # 예측 결과 출력
    predictions = nn.predict(X)
    for input_val, pred in zip(X, predictions):
        print(f"입력: {input_val}, 예측값: {pred[0]:.4f}")
    

    # 사용 가능한 데이터셋 확인
    print("사용 가능한 데이터셋:", dm.list_datasets())
