import cv2
import os


# 辅助函数：确保输出目录存在
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 辅助函数：保存裁剪并调整大小的人脸
def save_face_image(output_dir, face, count, size=(92, 112)):
    face_resized = cv2.resize(face, size)  # 调整人脸到指定大小
    filename = os.path.join(output_dir, f"u{count+1}.pgm")  # 生成文件名
    cv2.imwrite(filename, face_resized)  # 保存图片
    print(f"保存图片: {filename}")


# 检测人脸并保存
def capture_and_save_faces(output_dir, haar_cascade_path, num_images=100, camera_index=0):
    # 创建 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    cap = cv2.VideoCapture(camera_index)
    ensure_directory(output_dir)

    count = 0
    while count < num_images:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret:
            print("无法捕获摄像头视频")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]  # 裁剪出人脸区域
            save_face_image(output_dir, face, count)  # 保存人脸
            count += 1

            # 在原图上绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 显示当前图像和检测结果
            cv2.imshow('Capturing Faces', frame)
            cv2.waitKey(1000)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 检查是否已经捕获到指定数量的图像
        if count >= num_images:
            print(f"已拍摄 {num_images} 张图片")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    OUTPUT_DIR = "user"  # 存储人脸图片的文件夹
    HAAR_CASCADE_PATH = r"D:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"  # Haar 分类器路径
    NUM_IMAGES = 100  # 要拍摄的图片数量

    # 调用函数，开始捕捉并保存人脸
    capture_and_save_faces(OUTPUT_DIR, HAAR_CASCADE_PATH, NUM_IMAGES)
