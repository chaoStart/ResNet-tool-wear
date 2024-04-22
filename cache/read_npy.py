# 卞庆朝
# 开发时间：2023/7/3 10:53
import numpy as np

# 从 .npy 文件中加载数组
array = np.load('G:\pycharm\PHM_regression\cache\RUL_X.npy')
array1 = np.load('../cache/RUL_Y.npy')
array2 = np.load('./all_res_dat_num_5000.npy')
# 合并数组
# merged_array = np.concatenate((array,array1,array2), axis=0)
# 打印数组
print(type(array))
print(array.shape)
print(type(array1))
print(array1.shape)
print(type(array2))
print(array2.shape)
print(array2[:,0])
# 合并后并且交换第二维和三维的顺序
# 交换第二和第三维度
# new_arr = np.transpose(merged_array, (0, 2, 1))
# print(new_arr.shape)
# # 保存数组
# np.save('new_arr.npy', new_arr)