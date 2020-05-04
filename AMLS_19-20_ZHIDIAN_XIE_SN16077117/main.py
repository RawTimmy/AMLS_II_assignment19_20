from A import model_4a, preprocess_4a
from B import model_4b, preprocess_4b

# Data Preprocessing
pre_4a = preprocess_4a.preprocess()
pre_4b = preprocess_4b.preprocess()

train_X_4a, train_Y_4a, test_X_4a, test_Y_4a = pre_4a.preprocess_4a()
train_X_4b, train_Y_4b, test_X_4b, test_Y_4b = pre_4b.preprocess_4b()

# Task 4A
model_bilstm = model_4a.utils_4a()
train_acc_4a, test_acc_4a = model_bilstm.process_4a(train_X_4a, train_Y_4a, test_X_4a, test_Y_4a)

# Task 4B
model_bilstm = model_4b.utils_4b()
train_acc_4b, test_acc_4b = model_bilstm.process_4b(train_X_4b, train_Y_4b, test_X_4b, test_Y_4b)

print("==========Results: Training Accuracy, Test Accuracy==========")
print('TA_train:{};\nTA_test:{};\nTB_train:{};\nTB_test:{};'.format(train_acc_4a,test_acc_4a,
                                                                    train_acc_4b,test_acc_4b))
