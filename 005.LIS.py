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