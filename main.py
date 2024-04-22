from PHM_regression.dataSet import auto_stride_dataset,wavelet_dataset
from PHM_regression.model import build_simple_rnn_model,build_multi_input_main_residual_network
from keras.callbacks import TensorBoard
import numpy as np
import h5py

a = auto_stride_dataset()
y = a.get_all_loc_y_sample_data()
x = a.get_all_loc_x_sample_data()
print('y的type:',y.shape)#输出的y---->(945,3)
print('x的type:',x.shape)#输入的x--->(945,5000,7)
data = wavelet_dataset()
a2,d2,d1 = data.gen_x_dat()
# y = data.gen_y_dat()
# y = data.get_rul_dat()
import random
index = [i for i in range(len(y))]
print('index的len,index表示[0,1,2,...945]:',len(index),index)
random.shuffle(index)
y = y[index]
a2, d2, d1 = a2[index],d2[index],d1[index]
print('第一个形状是y的，剩余的是a2,d2,d1:',y.shape,a2.shape,d2.shape,d1.shape)
# y,a2,d2,d1: (945, 3) (945, 1255, 7) (945, 1255, 7) (945, 2503, 7)
print('d2.shape[1]的形状和大小',d2.shape[1])

for i in [20,15,10,5,35]:
    print('i是多少:',i)
    DEPTH = i
    log_dir = 'resiual_RUL_logs/'
    train_name = 'residual_logs_adam_adjust_kernel_depth_%s_shuffle_manually_LOCAL' % (DEPTH)
    model_name = '%s.kerasmodel' % (train_name)
    predict = True
    # predict = False
    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name)
        # model = build_multi_input_main_residual_network(32, a2.shape[1], d2.shape[1], d1.shape[1], 7, 1,loop_depth=DEPTH)
        model = build_multi_input_main_residual_network(32, a2.shape[1], d2.shape[1],
                                                        d1.shape[1], 7, 3,loop_depth=DEPTH)
        # model = build_simple_rnn_model(5000,7,3)
        print('Model has been established.')
        model.fit([a2, d2, d1],y, batch_size=16, epochs=2, callbacks=[tb_cb], validation_split=0.2)
        # model.save(model_name)
        #保存Keras模型
        import h5py
        # model.save(model_name)
        # model.save("Kerasmodel1.h5")
        print('模型保存成功ending~~')

    else:
        PRED_PATH = 'Y_PRED'
        a = auto_stride_dataset()
        y = a.get_all_loc_y_sample_data()
        x = a.get_all_loc_x_sample_data()
        print('y真实数据的形状:',y.shape)
        print('y真实数据打印前面（315,3）数据:\n',y[:315,:].shape)
        print('y真实数据打印前面（315,3）数据:\n',y[:315,:])
        #这里调用小波变换wavelet,由于wavelet自动依次调用了x = a.get_all_loc_x_sample_data()
        # 和y = a.get_all_loc_y_sample_data();所以又打印了一遍storage_path的路径。。。
        data = wavelet_dataset()
        a2, d2, d1 = data.gen_x_dat()
        print('a2,d2,d1三个的形状:',type(a2),type(d1),type(d2))
        try:
            import numpy as np
            y_pred = np.load(PRED_PATH + '.npy')
        except Exception as e:
            print(e)
            from keras.models import load_model
            model = load_model(model_name)
            # model = load_model("Kerasmodel.h5")
            y_pred = model.predict([a2, d2, d1])
            # y_pred = model.predict(x)
            print('y_pred预测值的形状大小:',y_pred.shape,y_pred)
            np.save(PRED_PATH, y_pred)
        # print('y_pred.shape的类型和大小~:',type(y_pred),y_pred.shape)

        json_data = {}
        json_data["predicted"] = y_pred.tolist()
        json_data["real"] = y.tolist()
        import json
        print(json.dumps(json_data))
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = 'SimHei'
        for i in range(3):
            for j in range(3):
                # plt.subplot('31%s'%(j+1))
                fig = plt.figure()
                print("%s tool @ %s teeth" %(i+1,j+1))
                plt.title('No.%s tool wear' % (i + 1))
                plt.plot(y[j * 315:(j + 1) * 315, i], label=u'actual')
                plt.plot(y_pred[j * 315:(j + 1) * 315, i], label=u'predict')
                plt.ylabel('tool wear ($\mu m$)')
                plt.xlabel('number')
                plt.legend()
                plt.savefig("%s_tool_@_%s_teeth_ZH.svg"%(i+1,j+1))
                plt.show()
    break
    # if True:
    #     bais = y_pred - y
    #     print(bais.shape)
    #     import seaborn as sns
    #     for i in range(3):
    #         sns.set(color_codes=True)
    #         sns.distplot(bais[:, i], kde=True)
    #         print(bais[:, i].shape)
    #         print('%' * 20)
    #         plt.tight_layout()
    #         # plt.show()
    #         # print(np.arange(1,946,1).shape,bais[:,i].shape)
    #         # n, bins, patches = plt.hist(bais[:,i],'auto',normed=1, facecolor='blue', alpha=0.5)
    #         # # curve
    #         # #y = mlab.normpdf(bins, mu, sigma)
    #         # #plt.plot(bins, y, 'r--')
    #         # plt.xlabel('Error')
    #         # plt.ylabel('Frequency')
    #         # plt.legend()
    #         # plt.show()
    #     plt.rcParams['font.sans-serif'] = ['PingFang SC']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # break






