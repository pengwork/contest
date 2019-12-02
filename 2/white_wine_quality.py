import pandas as pd 
import tensorflow as tf

csv_column_names = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates",  "alcohol", "quality"]
data_train = pd.read_csv('./train.txt', sep = ";", names = csv_column_names)
data_test = pd.read_csv('./test.txt', sep = ";", names = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates",  "alcohol"])
data_train.head()

train_x, train_y = data_train, data_train.pop('quality')
test_x = data_test
train_y = train_y - 1
train_x.head()

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print(my_feature_columns)
    
classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns,
                                      hidden_units=[1024, 512, 256],
                                       n_classes=10,                                       
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.01,
      l1_regularization_strength=0.001
    ))

def train_func(train_x, train_y):
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    dataset = dataset.shuffle(1000).repeat().batch(100)
    return dataset
classifier.train(input_fn = lambda:train_func(train_x, train_y),
                steps= 10000)
              
def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset
    
predict_arr = []
predictions = classifier.predict(input_fn=lambda:eval_input_fn(test_x,labels=None,batch_size=100))
for predict in predictions:
    predict_arr.append(predict['probabilities'].argmax())


output = pd.DataFrame(predict_arr)
output.to_csv("result.csv", index = True, header =False)
