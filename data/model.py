'''
Author: AtlasCodex wenlin.xie@outlook.com
Date: 2024-07-02 00:12:31
LastEditors: AtlasCodex wenlin.xie@outlook.com
LastEditTime: 2024-07-17 15:56:44
FilePath: /ticket/data/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

class LotteryPredictionModel:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_gpu()
        self.model = None
        self.scaler = MinMaxScaler()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_gpu(self):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def create_sequences(self, data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    def prepare_data(self, matrix, lottery_type):
        window_size = self.config['model_params'][lottery_type]['window_size']
        X, y = self.create_sequences(matrix, window_size)
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_shape, lottery_type):
        model_params = self.config['model_params'][lottery_type]
        model = Sequential([
            LSTM(model_params['lstm_units_1'], activation='relu', return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(model_params['dropout_rate']),
            LSTM(model_params['lstm_units_2'], activation='relu'),
            BatchNormalization(),
            Dropout(model_params['dropout_rate']),
            Dense(model_params['output_units'], activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=model_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, X_val, y_train, y_val, lottery_type):
        model_params = self.config['model_params'][lottery_type]
        input_shape = (model_params['window_size'], X_train.shape[-1])
        self.model = self.build_model(input_shape, lottery_type)
        model_save_path = model_params['bestmodel_paths']

        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
        # checkpoint = ModelCheckpoint(f'best_{lottery_type}_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=model_params['patience'], restore_best_weights=True)

        batch_size = model_params['batch_size']
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        history = self.model.fit(
            train_dataset,
            epochs=model_params['epochs'],
            validation_data=val_dataset,
            callbacks=[checkpoint, early_stopping]
        )
        return history

    def evaluate_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"训练集准确率: {train_acc:.4f}")
        print(f"验证集准确率: {val_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")

    def save_model(self, lottery_type):
        model_params = self.config['model_params'][lottery_type]
        final_model_save_path = model_params['finalmodel_paths']
        self.model.save(final_model_save_path)

    def run(self, matrix, lottery_type):
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(matrix, lottery_type)
        history = self.train_model(X_train, X_val, y_train, y_val, lottery_type)
        self.evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test)
        self.save_model(lottery_type)
        return history

# 使用示例
if __name__ == "__main__":
    model = LotteryPredictionModel()
    # 假设matrix是您的输入数据
    matrix = np.random.rand(1000, 49)  # 这里只是一个示例，请使用实际的数据
    model.run(matrix, 'ssq')

