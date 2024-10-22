import os

def create_csv(base_path, output_path):
    with open(output_path, "w") as f:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".pgm"):
                    label = os.path.basename(root)
                    img_path = os.path.join(root, file)
                    f.write(f"{img_path},{label}\n")

if __name__ == "__main__":
    BASE_PATH = "dataset"
    OUTPUT_PATH = "dataset/data.csv"
    create_csv(BASE_PATH, OUTPUT_PATH)
    print("CSV 文件生成完成")
