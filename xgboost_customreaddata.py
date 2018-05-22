import scipy
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_data(path):
    y = []
    row = []
    col = []
    values = []
    r = 0 #首行
    for d in open(path):
        d = d.strip().split() #以空格分开
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(':')
            row.append(r)
            col.append(int(key))
            values.append(float(value))
        r += 1
    # csr_matrix两种用法：
    # csr_matrix((data, (row_ind, col_ind)), [shape = (M, N)])
    # 　　where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
    # #csc_matrix((data, indices, indptr), [shape=(M, N)])
    #values: 稀疏矩阵中元素
    #indices: 稀疏矩阵非0元素对应的列索引值所组成数组
    #indptr: 第一个元素0，之后每个元素表示稀疏矩阵中每行元素（非零元素）个数累计结果
    x = scipy.sparse.csr_matrix((values, (row, col))).toarray()
    y = np.array(y)
    return x, y



def startjob():
    x_train, y_train = read_data('./data/agaricus.txt.train')
    x_test, y_test = read_data('./data/agaricus.txt.test')

    #Logistic回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    print('Logistic回归正确率: ', accuracy_score(y_test, y_hat))

    #XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objectve':'multi:softmax', 'num_class':3}
    # param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objectve': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误数目: \t%4d' % error)
    print('错误率: \t%.5f%%' % (100 * error_rate))

if __name__ == '__main__':
    startjob()