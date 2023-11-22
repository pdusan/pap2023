# Copyright (c) 2023 Dušan Popović
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

matrix = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]
]

m = len(matrix)
n = len(matrix[0])

for d in range(m+n-1):
    for i in range (min(d, m-1)+1, max(0, d-n+1)):
        j = d - i
        print(matrix[i][j])