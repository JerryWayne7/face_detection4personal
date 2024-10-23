import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 读取数据
data = pd.read_csv('dataset/data.csv', header=None)
print(data.head())

# 假设数据的第一列是标签，剩余的是图像数据
labels = data.iloc[:, 2].values
path = data.iloc[:, 0].values
images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in path]
print(labels)
print(path)

# 将图像数据转换为适合模型训练的格式
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    image = cv2.resize(image, (64, 64))  # 调整图像大小
    return image

images = np.array([preprocess_image(img) for img in images])
images = images.reshape(-1, 64, 64, 1)  # 调整图像形状以适应CNN输入
labels = to_categorical(labels)  # 将标签转换为one-hot编码

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 定义数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 定义CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(labels.shape[1], activation='softmax')
])

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型，使用数据增强
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
print("模型训练完成")

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 保存训练好的模型
model.save('cnn_model.h5')
print("模型已保存至 cnn_model.h5")