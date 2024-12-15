import numpy as np
import os
import time

# 도넛의 점 생성 (파라메트릭)
def create_donut_points(num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)

    R = 2  # 도넛의 큰 반지름
    r = 1  # 도넛의 작은 반지름

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    return np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)  # (N, 3) 배열

# 회전 행렬
def get_rotation_matrix(theta, phi, psi):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)],
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx  # Z -> Y -> X 순으로 회전

# 투영
def project_to_2d(points):
    projection_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])
    return points @ projection_matrix.T

# 도넛 애니메이션
def animate_donut():
    points = create_donut_points()
    theta, phi, psi = 0, 0, 0  # 초기 회전 각도

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # 화면 초기화

        # 회전
        rotation_matrix = get_rotation_matrix(theta, phi, psi)
        rotated_points = points @ rotation_matrix.T

        # 투영
        projected_points = project_to_2d(rotated_points)

        # 터미널 출력 (간단히 좌표를 출력)
        grid = np.full((40, 80), ' ')
        for x, y in projected_points:
            xi = int(20 + x * 10)  # 화면 좌표로 변환
            yi = int(40 + y * 20)
            if 0 <= xi < 40 and 0 <= yi < 80:
                grid[xi, yi] = '@'

        # 출력
        for row in grid:
            print(''.join(row))

        # 회전 각도 업데이트
        theta += 0.1
        phi += 0.05
        psi += 0.03
        time.sleep(0.03)

# 실행
animate_donut()
