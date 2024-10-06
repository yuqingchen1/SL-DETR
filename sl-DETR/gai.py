import os

# 定义类别与数字的映射关系
category_to_number = {
    "bridge": 0,
    # "plane": 0,
    # "baseball-diamond": 1,
    # "bridge": 2, "ground-track-field": 3, "small-vehicle": 4,
    # "large-vehicle": 5, "ship": 6, "tennis-court": 7,
    # "basketball-court": 8, "storage-tank": 9, "soccer-ball-field": 10,
    # "roundabout": 11, "harbor": 12, "swimming-pool": 13,
    # "helicopter": 14,
    # 根据实际需要定义更多类别和数字映射
}


def update_txt_file(input_file, output_file):
    """更新单个txt文件的第9列与第10列数据，使其保持一致"""
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w',
                                                                      encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue  # 跳过空行

            # 按空格或制表符分割行数据
            columns = line.strip().split()

            if len(columns) < 10:
                print(f'跳过无效行: {line.strip()}')
                continue  # 跳过列数不足的行

            category = columns[8]  # 第9列为类别名
            number = columns[9]  # 第10列为数字

            # 检查类别与数字是否匹配
            if category in category_to_number:
                expected_number = category_to_number[category]
                if int(number) != expected_number:
                    print(f'更新类别 {category} 的数字，从 {number} 变为 {expected_number}')
                    columns[9] = str(expected_number)  # 更新数字

            # 将处理后的行写入输出文件
            outfile.write(' '.join(columns) + '\n')


def process_all_txt_files(input_dir, output_dir):
    """处理输入目录中的所有txt文件"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有txt文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            print(f'正在处理文件: {filename}')
            update_txt_file(input_file, output_file)
            print(f'已处理并保存到: {output_file}')


if __name__ == "__main__":
    input_dir = '/home/yu/桌面/RHINO-main/data/split_ss_ghl/train/annfiles'  # 包含多个txt文件的输入目录
    output_dir = '/home/yu/桌面/RHINO-main/data/split_ss_ghl/train1/annfiles'  # 输出目录，用于保存处理后的txt文件
    process_all_txt_files(input_dir, output_dir)
