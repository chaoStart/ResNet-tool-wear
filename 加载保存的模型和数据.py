# 卞庆朝
# 开发时间：2024/4/7 14:45
import numpy as np
from keras.models import load_model
from  dataSet import auto_stride_dataset
DEPTH=20
train_name = 'residual_logs_adam_adjust_kernel_depth_%s_shuffle_manually_LOCAL' % (DEPTH)
model_name = '%s.kerasmodel' % (train_name)
model = load_model(model_name)
PRED_PATH = 'Y_PRED'
y_pred = np.load(PRED_PATH + '.npy')

a = auto_stride_dataset()
y = a.get_all_loc_y_sample_data()
print('y的真实数据形状：',y.shape)
# y=y[:315,:]#这里是C6的磨损数据
# y=y[315:630,:]#这里是C4的磨损数据
# y=y[630:945,:]#这里是C1的磨损数据
print('y_pred的预测数据形状：',y_pred.shape)

# json_data = {}
# json_data["predicted"] = y_pred.tolist()
# json_data["real"] = y.tolist()

import json
# print(json.dumps(json_data))
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文标题
plt.rcParams['font.family'] = 'Times New Roman'
xx1 = range(0, 315)
# for j in range(3):
#     for i in range(3):
#         # plt.subplot('31%s'%(j+1))
#         fig = plt.figure()
#         print("%s tool @ %s teeth" % (j + 1, i + 1))
#         print('No.%s tool wear' % (j + 1))
#         plt.title('No.%s tool wear' % (j + 1))
#         # plt.plot(y[j * 315:(j + 1) * 315, i], label=u'actual')
#         # plt.plot(y_pred[j * 315:(j + 1) * 315, i], label=u'predict')
#         # plt.ylabel('tool wear ($\mu m$)')
#         # plt.xlabel('number')
#         # plt.legend()
#         # plt.savefig("%s_tool_@_%s_teeth_ZH.svg" % (i + 1, j + 1))
#         # plt.show()
#         print('i是: %s和j是: %s'%(i+1,j+1))
#         y_actual=y[j * 315:(j + 1) * 315, i]
#         y_pre=y_pred[j * 315:(j + 1) * 315, i]
#         # 计算实际值和预测值的差值loss
#         loss=np.abs(y_actual-y_pre)
#         # print('计算实际和预测的误差：',loss.shape,'\n',loss)
#         plt.plot(xx1, y_pre, color=(0, 106/255, 174/255), markerfacecolor='none',markeredgewidth=0,label='proposed')
#         plt.plot(xx1, y_actual, color='r', markerfacecolor='none',markeredgewidth=0, label='Ture data')
#         plt.hist(xx1,weights=loss,color=(62/255, 75/255, 168/255), edgecolor='none',bins=315,label='Error')
#         plt.xlabel('Cut Number', fontsize=20)
#         plt.ylabel('Tool wear ($\mu m$)', fontsize=20)
#         # 手动设置标签位置为左上角
#         plt.legend(loc='upper left')
#         # plt.savefig("%s_tool_@_%s_teeth_ZH.svg" % (j + 1, i + 1))
#         plt.show()

# 对每列求平均值
y_pred=y_pred[:315,:]
print('C6的y预测 磨损均值:',y_pred.shape)
y_pred = np.mean(y_pred, axis=1,keepdims=True)
print('C6 三个刀具磨损值的均值y_pred:',y_pred.shape)
y=y[:315,:]
print('C6的y真实 磨损均值:',y.shape)
y= np.mean(y, axis=1,keepdims=True)
print('C6 三个刀具磨损值的均值y:',y.shape)
# 计算实际值和预测值的差值loss
loss=np.abs(y-y_pred)
plt.title('C6工况下均值预测', fontproperties='SimHei' )
plt.plot(xx1, y_pred, color=(0, 106/255, 174/255), markerfacecolor='none',markeredgewidth=0,label='proposed')
plt.plot(xx1, y, color='r', markerfacecolor='none',markeredgewidth=0, label='Ture data')
plt.hist(xx1,weights=loss,color=(62/255, 75/255, 168/255), edgecolor='none',bins=315,label='Error')
# plt.plot(xx1,loss,color=(62/255, 75/255, 168/255),label='Error')
plt.xlabel('Cut Number', fontsize=20)
plt.ylabel('Tool Wear(μm)', fontsize=20)
# plt.ylabel(r'Average wear$\mu m$')
# 手动设置标签位置为左上角
plt.legend(loc='upper left')
plt.savefig("残差网络模型.svg")
plt.show()