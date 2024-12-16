import os
import random
import json
import requests
import csv
import re
import json
import os
import os
import csv
import subprocess

def analyze_directory(root_directory, sample_folder_count=1, sample_file_count=5):
    # 用于存储最终的结果
    result = {
        "root_directory": root_directory,
        "a_level_summary": {
            "total_a_folders": 0,
            "example_a_folders": [],
        },
        "sampled_a_folders": []
    }

    # 1. 计算根目录下的 A 级文件夹数量并打印
    a_level_folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]
    result["a_level_summary"]["total_a_folders"] = len(a_level_folders)
    result["a_level_summary"]["example_a_folders"] = a_level_folders[:5]

    # 2. 随机采样 A 级文件夹
    sampled_a_folders = random.sample(a_level_folders, min(sample_folder_count, len(a_level_folders)))

    for sampled_a_folder in sampled_a_folders:
        sampled_a_path = os.path.join(root_directory, sampled_a_folder)
        sampled_a_info = {
            "a_folder_name": sampled_a_folder,
            "directory_tree": [],
            "sampled_files": []
        }

        # 3. 穷尽 A 级文件夹下的目录树
        file_list = []
        for root, dirs, files in os.walk(sampled_a_path):
            # 获取当前路径的相对路径和层级
            relative_root = os.path.relpath(root, sampled_a_path)
            folder_level = len(relative_root.split(os.sep)) if relative_root != "." else 0

            # 保存目录树信息
            sampled_a_info["directory_tree"].append({
                "level": folder_level,
                "folder_name": os.path.basename(root),
                "sub_folders": dirs,
                "file_count": len(files)
            })

            # 收集文件地址
            file_list.extend([os.path.join(root, f) for f in files])

        # 4. 随机采样末端文件
        sampled_files = random.sample(file_list, min(sample_file_count, len(file_list)))
        sampled_a_info["sampled_files"] = sampled_files

        # 添加到结果中
        result["sampled_a_folders"].append(sampled_a_info)

    # 将结果格式化为 JSON 并打印
    formatted_result = json.dumps(result, indent=4, ensure_ascii=False)
    print(formatted_result)

    # 可选择将结果保存到文件
    output_file = os.path.join(root_directory, "directory_analysis.json")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_result)
    print(f"\n结果已保存到: {output_file}")



def generate_metadata(root_directory, your_api_key=None):
    # DeepSeek API 的 URL 和 API 密钥
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    llm_api = None
    if your_api_key is None:
        if os.path.exists(r"D:\REPO\PreProcPipe\config.py"):
            from config import API_KEY
            llm_api = API_KEY
            print("API 密钥已从 config.py 中读取。")
    else:
        llm_api = your_api_key

    # 读取 JSON 文件
    with open(os.path.join(root_directory, "directory_analysis.json"), "r") as f:
        json_input = f.read()
    print("正在指使LLM生成代码...")
    # 构建请求数据
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一名熟练的数据科学家，善于解析复杂的文件目录并生成元数据表格。"
                    "你的任务是帮助用户分析医学影像数据集，并根据采样的文件结构生成metadata.csv。"
                    "你只需要输出带有恰当注释的python代码即可，多余的信息不输出。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"我正在浏览一个医学影像数据集，它的根目录为：{json.loads(json_input)['root_directory']}。\n"
                    "这个数据集包含若干影像文件（可能包括多模态文件、单模态文件和掩码文件）。\n"
                    "我采样了一些子文件夹（记为 A 级文件夹）以及其中的 B/C 级文件夹，目录树和采样文件的信息如下：\n"
                    f"{json_input}\n"
                    "我需要你：\n"
                    "1. 分析文件命名的规律，判断是否存在多模态文件或掩码文件，分析出他们之间配对的关系，比如命名可能有相同的地方，或者用后缀区分了图像与掩码。\n"
                    "2. 根据这些规律生成构建 metadata.csv 的 Python 代码。\n"
                    "3. 输出的代码应该以根目录为输入，生成的 csv 应保存在根目录下，csv 的列包括 sample_id（若没有明显 id，则直接用数字序号）、各模态的文件地址（如 flair_path, t1_path 等，若没有明显的多模态特征那么记为image_path）、以及掩码地址（若不存在则为空）。\n"
                )
            }
        ],
        "stream": False
    }

    # 发送请求
    headers = {
        "Authorization": f"Bearer {llm_api}",
        "Content-Type": "application/json"
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        try:
            # 打印模型输出的原始内容
            print("模型输出的原始内容：")
            model_output = result["choices"][0]["message"]["content"]
            print(model_output)

            # 保存 LLM 输出到 result.json
            with open("result.json", "w") as f:
                f.write(model_output)
            print("LLM 的响应内容已保存到 result.json 文件中。")

            # 尝试从 LLM 的输出中提取生成的代码
            code_match = re.search(r"```python(.*?)```", model_output, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
                with open(os.path.join(root_directory, "generate_metadata.py"), "w") as f:
                    f.write(extracted_code)
                print("生成的 Python 代码已保存到 generate_metadata.py 文件中。")
            else:
                print("未检测到有效的 Python 代码块，请手动检查 LLM 输出。")
        except Exception as e:
            print(f"解析响应内容时发生错误：{e}")
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.text)



def execute_metadata_script(root_directory):
    metadata_file = os.path.join(root_directory, "metadata.csv")
    script_file = os.path.join(root_directory, "generate_metadata.py")

    # 检查 generate_metadata.py 是否存在
    if not os.path.exists(script_file):
        print(f"脚本 {script_file} 不存在，请确保文件已正确生成。")
    else:
        # 执行 generate_metadata.py 脚本
        print(f"正在执行 {script_file}...")
        result = subprocess.run(["python", script_file], capture_output=True, text=True)

        # 检查执行结果
        if result.returncode == 0:
            print(f"{script_file} 执行成功！")
        else:
            print(f"{script_file} 执行失败！")
            print(f"错误输出：\n{result.stderr}")

        # 检查 metadata.csv 是否存在
        if os.path.exists(metadata_file):
            print(f"metadata.csv 文件已生成，路径为：{metadata_file}")

            # 打印 metadata.csv 的前 5 行
            try:
                with open(metadata_file, "r") as f:
                    reader = csv.reader(f)
                    print("metadata.csv 的前 5 行内容：")
                    for i, row in enumerate(reader):
                        print(row)
                        if i == 4:  # 打印前 5 行
                            break
            except Exception as e:
                print(f"读取 metadata.csv 时发生错误：{e}")
        else:
            print("metadata.csv 文件未生成，请检查脚本逻辑和根目录路径。")

def metadata_sanity_check(root_directory):
    metadata_file = os.path.join(root_directory, "metadata.csv")
    
    try:
        with open(metadata_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    if '_path' in key:
                        if not value:
                            print(f"空值: {key} 在 sample_id {row['sample_id']} 中为空")
                        else:
                            full_path = os.path.join(root_directory, value)
                            if not os.path.exists(full_path):
                                print(f"路径无效: {key} 在 sample_id {row['sample_id']} 中指向 {full_path}")
                            else:
                                # print(f"路径有效: {key} 在 sample_id {row['sample_id']} 中指向 {full_path}")
                                pass
    except Exception as e:
        print(f"读取 metadata.csv 时发生错误: {e}")
        print("看起来有错误，你可以手动查看 metadata.csv 是否正确。")

    


if __name__ == "__main__":
    # 1. 分析文件目录结构
    root_directory = r"D:\REPO\PreProcPipe\BraTS2021_Training_Data" # 填写数据集根目录，一定要是绝对路径

    analyze_directory(root_directory=root_directory, sample_folder_count=5, sample_file_count=10) # 可以通过增加 sample_folder_count 和 sample_file_count 来提高成功率

    # 2. 生成 metadata.csv 的 Python 代码
    generate_metadata(root_directory=root_directory, your_api_key=None) # 填写你的API_KEY
    # 推荐你去DeepSeek官网注册一个账号，然后在个人中心获取API_KEY，他们会给你一辈子用不完的额度，输入格式为API = "sadasdasdwqeqwe2"

    # 3. 执行生成的代码并检查 metadata.csv
    execute_metadata_script(root_directory=root_directory)

    # 4. 检查 metadata.csv 的正确性
    metadata_sanity_check(root_directory=root_directory)


