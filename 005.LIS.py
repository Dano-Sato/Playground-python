from bisect import bisect_left

def longest_increasing_subsequence(arr):
    if not arr:
        return []
    
    # dp[i]는 길이가 i+1인 증가 부분 수열이 끝날 수 있는 가장 작은 숫자
    dp = []
    last_indices = []
    # prev[i]는 arr[i]가 속한 LIS에서 이전 숫자의 인덱스
    prev = [-1] * len(arr)
    # dp에서의 위치를 저장
    pos = [-1] * len(arr)
    
    for i, num in enumerate(arr):
        # 이진 탐색으로 num이 들어갈 위치 찾기
        idx = bisect_left(dp, num)
        
        if idx == len(dp):
            dp.append(num)
        else:
            dp[idx] = num
            
        pos[i] = idx
        prev[i] = -1 if idx == 0 else last_indices[idx-1]
        
        # 각 길이별 마지막 인덱스 갱신
        if idx >= len(last_indices):
            last_indices.append(i)
        else:
            last_indices[idx] = i
            
    # LIS 재구성
    lis = []
    curr_idx = last_indices[len(dp)-1]
    while curr_idx != -1:
        lis.append(arr[curr_idx])
        curr_idx = prev[curr_idx]
    
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