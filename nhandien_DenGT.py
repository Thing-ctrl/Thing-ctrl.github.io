import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import cv2
import matplotlib.pyplot as plt

# Thông số cơ bản
IMG_HEIGHT, IMG_WIDTH = 32, 32
SO_LOAI_BIEN_BAO = 43  # GTSRB có 43 loại biển báo giao thông

# Danh sách tên biển báo
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

# Đường dẫn đến các thư mục dữ liệu
thu_muc_meta = 'Meta'   # Đường dẫn tới thư mục Meta
thu_muc_train = 'Train' # Đường dẫn tới thư mục Train
thu_muc_test = 'Test'   # Đường dẫn tới thư mục Test

# Tạo các mảng lưu ảnh và nhãn
anh = []
nhan = []

# Load ảnh từ thư mục Train và resize
for loai_bien_bao in range(SO_LOAI_BIEN_BAO):
    thu_muc_loai = os.path.join(thu_muc_train, str(loai_bien_bao))
    if os.path.exists(thu_muc_loai):
        for tep_anh in os.listdir(thu_muc_loai):
            duong_dan_anh = os.path.join(thu_muc_loai, tep_anh)
            img = cv2.imread(duong_dan_anh)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            anh.append(img)
            nhan.append(loai_bien_bao)

anh = np.array(anh)
nhan = np.array(nhan)

# Chuẩn hóa ảnh
anh = anh / 255.0

# Chia thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(anh, nhan, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
mo_hinh = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(SO_LOAI_BIEN_BAO, activation='softmax')
])

# Biên dịch mô hình
mo_hinh.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mo_hinh.summary()

# Huấn luyện mô hình
lich_su = mo_hinh.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=1)

# Đánh giá mô hình trên tập kiểm thử
y_du_doan = np.argmax(mo_hinh.predict(X_test), axis=1)
print("Độ chính xác:", accuracy_score(y_test, y_du_doan))
print(classification_report(y_test, y_du_doan))

# Lưu mô hình đã huấn luyện
mo_hinh.save('mo_hinh_bien_bao.h5')
print("Mô hình đã được lưu thành công!")

# Hàm dự đoán biển báo giao thông từ ảnh thực tế
def du_doan_bien_bao(duong_dan_anh):
    img = cv2.imread(duong_dan_anh)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch

    du_doan = np.argmax(mo_hinh.predict(img))
    print(f"Dự đoán biển báo: {class_names[du_doan]}")
    return class_names[du_doan]

# Hàm tải và tiền xử lý dữ liệu test
def load_test_data(data_dir):
    images = []
    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            images.append(image)

    images = np.array(images) / 255.0
    return images

# Đường dẫn đến tập test
test_dir = "Test"  # Thay đổi đường dẫn phù hợp với tập test của bạn

# Tải tập dữ liệu test
X_test = load_test_data(test_dir)

# Tải mô hình đã huấn luyện
model = load_model("mo_hinh_bien_bao.h5")

# Dự đoán trên tập test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Số lượng ảnh tối đa để hiển thị
max_images = min(36, len(X_test))  # Giới hạn hiển thị tối đa 36 ảnh

# Hiển thị ảnh và tên biển báo dự đoán
plt.figure(figsize=(15, 15))
for i in range(max_images):
    plt.subplot(6, 6, i+1)  # Sắp xếp ảnh trong lưới 6x6
    plt.imshow(X_test[i])
    plt.title(f"Dự đoán: {class_names[y_pred_classes[i]]}")
    plt.axis('off')  # Ẩn các trục tọa độ

plt.tight_layout()
plt.show()
