from bisect import bisect_left

def longest_increasing_subsequence(arr):
    if not arr:
        return []
    
    # dp[i]: 길이가 i+1인 증가 부분 수열의 마지막 숫자 중 가능한 가장 작은 값
    # 예: [3,1,4,2,5]에서 dp는 [1,2,5]가 됨 (길이 1,2,3인 수열의 최소 마지막 값들)
    dp = []
    
    # last_indices[i]: 길이가 i+1인 증가 부분 수열을 만들 때 사용된 원본 배열의 인덱스
    # LIS를 나중에 역추적할 때 사용
    last_indices = []
    
    # prev[i]: arr[i]로 끝나는 증가 수열에서 arr[i] 이전 숫자의 인덱스
    # 예: prev[4] = 1은 arr[4]가 arr[1] 다음에 온다는 의미
    prev = [-1] * len(arr)
    
    # pos[i]: arr[i]가 들어간 dp 배열에서의 위치 (0부터 시작)
    # 즉, arr[i]로 끝나는 증가 수열의 길이 - 1
    pos = [-1] * len(arr)
    
    for i, num in enumerate(arr):
        # 현재 숫자(num)가 들어갈 수 있는 위치를 이진 탐색으로 찾음
        # dp는 항상 정렬된 상태를 유지하므로 이진 탐색 가능
        idx = bisect_left(dp, num)
        
        if idx == len(dp):
            # num이 dp의 모든 수보다 크다면 새로운 길이의 수열 생성
            dp.append(num)
        else:
            # num이 더 작은 값이면 해당 위치의 값을 교체
            # 이는 후속 숫자들이 더 작은 값에 연결될 수 있게 해줌
            dp[idx] = num
            
        # 현재 숫자가 만드는 증가 수열의 길이를 저장
        pos[i] = idx
        
        # 현재 숫자의 이전 숫자 인덱스를 저장
        # idx가 0이면 이전 숫자가 없으므로 -1
        # 아니면 같은 길이-1 수열의 마지막 인덱스를 저장
        prev[i] = -1 if idx == 0 else last_indices[idx-1]
        
        # 현재 길이에 대한 마지막 인덱스 갱신
        if idx >= len(last_indices):
            last_indices.append(i)
        else:
            last_indices[idx] = i
    
    # LIS 재구성: 마지막 인덱스부터 시작해서 prev 배열을 따라가며 역순으로 수열 생성
    lis = []
    curr_idx = last_indices[len(dp)-1]  # 가장 긴 수열의 마지막 인덱스
    while curr_idx != -1:
        lis.append(arr[curr_idx])
        curr_idx = prev[curr_idx]
    
    # 역순으로 만들어진 수열을 다시 뒤집어서 반환
    return lis[::-1]

def count_lis(arr):
    if not arr:
        return 0, []
    
    n = len(arr)
    dp = []  # 길이별 최소값
    count = [1] * n  # 각 위치에서 끝나는 LIS의 개수
    last_indices = []
    
    for i, num in enumerate(arr):
        idx = bisect_left(dp, num)
        
        if idx == len(dp):
            dp.append(num)
        else:
            dp[idx] = num
            
        if idx > 0:
            # i번째 숫자로 끝나는 LIS의 개수 계산
            for j in range(i):
                if arr[j] < num and pos[j] == idx - 1:
                    count[i] += count[j]
                    
    # 최대 길이를 가진 모든 LIS의 개수 합산
    max_len = len(dp)
    total_count = sum(count[i] for i in range(n) if pos[i] == max_len - 1)
    
    return total_count, dp

def minimum_changes_to_lis(arr):
    n = len(arr)
    # 원하는 길이의 LIS를 만들기 위해 필요한 최소 변경 횟수
    dp = [[float('inf')] * (n + 1) for _ in range(n)]
    
    # 초기화: 길이 1인 수열은 변경 불필요
    for i in range(n):
        dp[i][1] = 0
    
    for length in range(2, n + 1):
        for i in range(length - 1, n):
            # i번째 숫자를 변경하지 않는 경우
            if i >= length - 1:
                for j in range(i):
                    if arr[i] > arr[j]:
                        dp[i][length] = min(dp[i][length], dp[j][length - 1])
            
            # i번째 숫자를 변경하는 경우
            for j in range(i):
                # j번째 숫자보다 큰 최소값으로 변경
                change_cost = 1
                dp[i][length] = min(dp[i][length], dp[j][length - 1] + change_cost)
    
    return min(dp[i][n] for i in range(n))

def box_stacking(boxes):
    """
    boxes: List of (width, height, depth) tuples
    returns: 최대로 쌓을 수 있는 상자의 높이
    """
    # 모든 가능한 회전 방향 생성
    rotations = []
    for w, h, d in boxes:
        # (base_width, base_depth, height)
        rotations.extend([
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
            (w, h, d)
        ])
    
    # 너비와 깊이를 기준으로 정렬
    rotations.sort(key=lambda x: x[0] * x[1], reverse=True)
    
    n = len(rotations)
    heights = [box[2] for box in rotations]  # 각 상자의 높이
    dp = heights[:]  # 각 상자까지의 최대 높이
    prev = [-1] * n  # 이전 상자 추적
    
    for i in range(1, n):
        for j in range(i):
            if (rotations[i][0] < rotations[j][0] and 
                rotations[i][1] < rotations[j][1]):
                current = heights[i] + dp[j]
                if current > dp[i]:
                    dp[i] = current
                    prev[i] = j
    
    # 최대 높이와 사용된 상자들 추적
    max_height = max(dp)
    stack = []
    curr = dp.index(max_height)
    while curr != -1:
        stack.append(rotations[curr])
        curr = prev[curr]
    
    return max_height, stack[::-1]
# 테스트
sequences = [
    [3, 10, 2, 1, 20, 30, 40, 10, 50, 25],
    [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15],
    [1, 5, 2, 3, 4, 9, 6, 7, 8]
]

for seq in sequences:
    lis = longest_increasing_subsequence(seq)
    print(f"원본 수열: {seq}")
    print(f"최장 증가 부분 수열: {lis}")
    print(f"길이: {len(lis)}\n")