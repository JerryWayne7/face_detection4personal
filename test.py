import os
import cv2


# 展示 user 文件夹中的所有 pgm 图片
def show_images():
    images = []
    for root, dirs, files in os.walk(r'D:\desktop\Human_Computer_Interaction\face_detection4personal\dataset\user'):
        for file in files:
            if file.endswith(".pgm"):
                img_path = os.path.join(root, file)
                images.append(cv2.imread(img_path))

    if not images:
        print("No images found.")
        return

    index = 0
    while True:
        cv2.imshow('image', images[index])
        print("size of image: ", images[index].shape)

        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('n'):  # Press 'n' for next image
            index = (index + 1) % len(images)
        elif key == ord('p'):  # Press 'p' for previous image
            index = (index - 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_images()
    print("展示完成")