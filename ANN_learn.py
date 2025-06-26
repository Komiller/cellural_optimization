import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
import json
from sklearn.metrics import mean_squared_error, r2_score


input_size = 81
batch_size = 27
def train_Cnum(json_file,result_file):
    """
    \
    :param json_file: path to file with trainig dataset .json {'inputs': [[<cell struts radii>],...],'outupts':[<stiffnes tensor element value>,...]}
    :param result_file: path to file where model saved
    :return:
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)  # load (без 's')
    x_train1 = np.array(data['inputs'])
    y_train1 = np.array(data['outputs'])

    x_train = tf.expand_dims(tf.cast(x_train1, tf.float32), axis=-1)
    y_train = tf.expand_dims(tf.cast(y_train1, tf.float32), axis=-1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(input_size)

    # Разделение на обучающую, валидационную и тестовую выборки
    train_size = int(0.7 * len(data['outputs']))  # 70% данных для обучения
    val_size = int(0.15 * len(data['outputs']))  # 15% данных для валидации
    test_size = len(data['outputs']) - train_size - val_size  # Остальные 15% для теста


    original_dataset=train_dataset
    train_dataset = original_dataset.take(train_size)
    val_dataset = original_dataset.skip(train_size).take(val_size)
    test_dataset = original_dataset.skip(train_size + val_size)

    train_dataset = train_dataset.batch(batch_size).repeat(10).cache()
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size).repeat(10).cache()
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.batch(1000).repeat(10).cache()
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


    model = keras.Sequential(
        [
            keras.layers.Dense(8, activation='tanh', input_shape=(3,)),
            keras.layers.Dense(1, activation='linear'),
        ]
    )
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(), metrics=[RootMeanSquaredError()],
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=300)





    test_dataset=list(test_dataset)
    test_x=np.array(test_dataset[0][0])
    xshape=test_x.shape
    test_x=test_x.reshape(xshape[0],3)
    y_true=np.array(test_dataset[0][1]).flatten()
    y_pred=model.predict(test_x, verbose=0)


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # RMSE (корень из среднеквадратичной ошибки)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R² (коэффициент детерминации)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    model.save(result_file)

train_Cnum('new_js/Ch11.json','all_weights/test_weight/model_11.keras')

