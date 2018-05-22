import numpy as np
import xgboost as xgb

#自定义损失函数的梯度和二阶导
def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0 - p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

def startjob():
    data_train = xgb.DMatrix('./data/agaricus.txt.train')
    data_test = xgb.DMatrix('./data/agaricus.txt.test')
    # print(data_train)
    # print(type(data_train))

    # param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'reg:logistic'}
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    bst = xgb.train(param,
                    data_train,
                    num_boost_round=n_round,
                    evals=watch_list,
                    obj=g_h,
                    feval=error_rate)
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0.5))
    errorrate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误数目: \t%4d' % error)
    print('错误率: \t%.5f%%' % (100 * errorrate))

if __name__ == '__main__':
    startjob()