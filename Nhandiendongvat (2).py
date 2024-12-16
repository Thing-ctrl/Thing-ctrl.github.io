import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Đường dẫn đến tập dữ liệu
dataset_path = "raw-img"

# Chuẩn bị tập dữ liệu
train_dir = os.path.join(dataset_path)
test_dir = os.path.join(dataset_path)

# Data Augmentation và tiền xử lý
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Xây dựng mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=test_generator
)
# Đánh giá tổng thể trên tập test
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Dự đoán và hiển thị
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)


# Vẽ biểu đồ huấn luyện và kiểm tra
plt.figure(figsize=(12, 5))
# Vẽ độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Vẽ hàm mất mát
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


dataset_path = 'raw-img/'
# Danh sách các lớp (tên thư mục)
class_labels = os.listdir(dataset_path)
# Số lượng ảnh cần hiển thị
num_images = 50
# Danh sách lưu ảnh và nhãn
selected_images = []
selected_labels = []
# Duyệt qua từng lớp, chọn ảnh ngẫu nhiên từ mỗi lớp
for class_label in class_labels:
    class_path = os.path.join(dataset_path, class_label)
    images_in_class = os.listdir(class_path)
    chosen_images = random.sample(images_in_class, min(5, len(images_in_class)))  # Chọn tối đa 5 ảnh/lớp

    for img_name in chosen_images:
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path, target_size=(128, 128))  # Resize ảnh về 128x128
        selected_images.append(img)
        selected_labels.append(class_label)

# Nếu chưa đủ 50 ảnh, chọn thêm ngẫu nhiên
while len(selected_images) < num_images:
    random_class = random.choice(class_labels)
    random_class_path = os.path.join(dataset_path, random_class)
    random_image = random.choice(os.listdir(random_class_path))
    img_path = os.path.join(random_class_path, random_image)
    img = load_img(img_path, target_size=(128, 128))
    selected_images.append(img)
    selected_labels.append(random_class)

# Tạo danh sách lưu dự đoán
predictions = []

# Dự đoán nhãn cho từng ảnh trong danh sách `selected_images`
for img in selected_images:
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    pred = model.predict(img_array)
    pred_label = np.argmax(pred)  # Lấy nhãn dự đoán
    predictions.append(pred_label)

# Hiển thị 50 ảnh với nhãn dự đoán và nhãn thực tế
plt.figure(figsize=(25, 20))
for i in range(num_images):
    plt.subplot(10, 5, i + 1)
    plt.imshow(selected_images[i])
    pred_label = predictions[i]  # Lấy nhãn dự đoán từ danh sách
    true_label = selected_labels[i]  # Nhãn thực tế
    plt.title(f"Pred: {class_labels[pred_label]}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()


def predict_image(image_path, model, class_indices):
    # Load ảnh và tiền xử lý
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    # Dự đoán
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = list(class_indices.keys())
    predicted_label = class_labels[predicted_class]

    # Hiển thị ảnh và kết quả
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Thử dự đoán với một ảnh tùy ý
image_path = "OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg"
predict_image(image_path, model, train_generator.class_indices)

