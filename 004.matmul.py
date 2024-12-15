import numpy as np
import time
import os

class Matrix:
    def __init__(self, data):
        self.data = np.array(data)
        
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        return Matrix(self.data @ other)
    
    def __rmatmul__(self, other):
        return Matrix(other) @ self
    
    def __str__(self):
        return str(self.data)

def create_cube():
    # 정육면체의 꼭지점 좌표
    points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ])
    # 꼭지점을 연결하는 선 (edges)
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # 아래면
        (4,5), (5,6), (6,7), (7,4),  # 윗면
        (0,4), (1,5), (2,6), (3,7)   # 수직 연결
    ]
    return points, edges

def get_rotation_matrix(theta, axis):
    """회전 행렬 생성"""
    if axis == 'x':
        return Matrix([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        return Matrix([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    else:  # z
        return Matrix([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

def project_to_2d(points, scale=10):
    """3D -> 2D 투영"""
    return np.array([[x * scale + 40, y * scale + 40] for x, y, z in points])

def draw_cube(points_2d, edges, canvas_size=(80, 80)):
    """캔버스에 큐브 그리기
    Args:
        points_2d: 2D로 투영된 큐브의 꼭지점 좌표들 [(x1,y1), (x2,y2), ...]
        edges: 선으로 연결할 꼭지점 쌍들의 리스트 [(start1,end1), (start2,end2), ...]
        canvas_size: 출력할 캔버스의 크기 (width, height)
    Returns:
        2D 문자 리스트로 표현된 캔버스
    """
    # 빈 캔버스 생성 (2D 문자 배열)
    canvas = [[' ' for _ in range(canvas_size[0])] for _ in range(canvas_size[1])]
    
    # 모든 edge(선)에 대해 반복
    for start, end in edges:
        # edge의 시작점과 끝점 좌표
        x1, y1 = map(int, points_2d[start])
        x2, y2 = map(int, points_2d[end])
        
        # Bresenham's line algorithm 구현
        # 이 알고리즘은 정수 연산만으로 직선을 그리는 효율적인 방법
        dx = abs(x2 - x1)  # x 변화량
        dy = abs(y2 - y1)  # y 변화량
        x, y = x1, y1      # 현재 그리는 위치
        
        # x, y 증가 방향 결정
        sx = 1 if x1 < x2 else -1  # x 방향
        sy = 1 if y1 < y2 else -1  # y 방향
        
        # x 변화가 y 변화보다 클 때 (기울기 < 1)
        if dx > dy:
            err = dx / 2  # 오차 누적용 변수
            while x != x2:  # 목표 x까지 반복
                # 캔버스 범위 내에 있을 때만 점 찍기
                if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                    canvas[y][x] = '#'
                    
                err -= dy  # 오차 누적
                # 오차가 임계값을 넘으면 y 좌표 조정
                if err < 0:
                    y += sy
                    err += dx
                x += sx  # x 좌표 증가
                
        # y 변화가 x 변화보다 클 때 (기울기 > 1)
        else:
            err = dy / 2  # 오차 누적용 변수
            while y != y2:  # 목표 y까지 반복
                # 캔버스 범위 내에 있을 때만 점 찍기
                if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                    canvas[y][x] = '#'
                    
                err -= dx  # 오차 누적
                # 오차가 임계값을 넘으면 x 좌표 조정
                if err < 0:
                    x += sx
                    err += dy
                y += sy  # y 좌표 증가
                
        # 마지막 점 찍기 (while 루프에서 마지막 점은 처리되지 않음)
        if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
            canvas[y][x] = '#'
    
    return canvas
def animate_cube():
    points, edges = create_cube()
    theta = 0
    
    while True:
        # 회전 행렬 생성 및 적용
        rot_x = get_rotation_matrix(theta, 'x')
        rot_y = get_rotation_matrix(theta * 0.5, 'y')
        rot_z = get_rotation_matrix(theta * 0.3, 'z')
        
        # 회전 적용
        rotated_points = points @ rot_x.data @ rot_y.data @ rot_z.data
        
        # 2D 투영
        points_2d = project_to_2d(rotated_points)
        
        # 그리기
        canvas = draw_cube(points_2d, edges)
        
        # 화면 클리어
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 출력
        for row in canvas:
            print(''.join(row))
            
        theta += 0.1
        time.sleep(0.05)

if __name__ == "__main__":
    animate_cube()