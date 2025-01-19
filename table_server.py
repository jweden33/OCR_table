from fastapi import FastAPI, File, UploadFile
import uvicorn
import subprocess
import tempfile
import os
import json
import re
import ast
import requests
from pdf2image import convert_from_path  # 导入pdf2image库
import uuid  # 导入uuid模块以生成唯一文件名
import shutil

app = FastAPI()

# 印章检测接口的URL
SEAL_RECOGNIZE_URL = 'http://h1337.iis.pub:24221/seal/recognize_seal'

# 自定义的临时文件存储目录
TEMP_FOLDER = './temp_files'

# 确保临时文件夹存在
os.makedirs(TEMP_FOLDER, exist_ok=True)

# 安全获取字典中的值
def safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default

@app.post("/process_image")
async def process_image(image: UploadFile = File(...)):
    temp_file_path = None  # 用于记录原始上传文件的路径
    temp_image_paths = []   # 用于记录转换后的图片路径（如果上传的是PDF）
    table_data = []
    seal_detection_result = None

    try:
        # 生成唯一的文件名，保留原始文件的扩展名
        original_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{original_extension}"
        temp_file_path = os.path.join(TEMP_FOLDER, unique_filename)

        # 将上传的文件内容写入临时文件
        with open(temp_file_path, 'wb') as temp_file:
            content = await image.read()
            temp_file.write(content)

        # 如果文件是PDF，则转换为图片用于表格检测
        if original_extension.lower() == '.pdf':
            try:
                images = convert_from_path(temp_file_path)
                for i, image_page in enumerate(images):
                    page_unique_filename = f"{uuid.uuid4().hex}_page_{i + 1}.png"
                    image_path = os.path.join(TEMP_FOLDER, page_unique_filename)
                    image_page.save(image_path, 'PNG')
                    temp_image_paths.append(image_path)
            except Exception as e:
                return {"error": f"PDF 转换为图片失败: {str(e)}"}
        else:
            temp_image_paths = [temp_file_path]  # 直接处理其他图片格式文件

    except Exception as e:
        # 如果保存文件失败，返回错误
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {"error": f"文件保存失败: {str(e)}"}

    try:
        # 进行表格检测
        for temp_image_path in temp_image_paths:
            # 运行表格检测命令
            command = [
                'python', 'table/predict_table.py',
                '--det_model_dir=inference/ch_PP-OCRv3_det_infer',
                '--rec_model_dir=inference/ch_PP-OCRv3_rec_infer',
                '--table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer_epo20',
                '--rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt',
                '--table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt',
                f'--image_dir={temp_image_path}',
                '--output=../output/table'
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            except Exception as e:
                return {"error": f"执行表格检测命令时出错: {str(e)}"}

            # 检查是否有日志信息混杂在标准输出中
            output = result.stdout
            # 使用正则表达式提取 [[...]] 部分
            pattern = r'(\[\[.*?\]\])'
            match = re.search(pattern, result.stdout, re.DOTALL)

            if match:
                json_output = match.group(1)
                try:
                    # 使用 ast.literal_eval 解析含单引号的JSON-like字符串
                    table = ast.literal_eval(json_output)
                    table_data.append(table)
                except (ValueError, SyntaxError) as e:
                    return {"error": f"清理后仍无法解析为JSON: {str(e)}", "cleaned_output": json_output}

        # 进行印章检测，基于原始上传的文件
        seal_detection_result = await recognize_seal(temp_file_path)

        # 如果印章识别失败，则将 seal_info 设置为 None
        seal_info = None
        if seal_detection_result and "result" in seal_detection_result:
            stamp_list = safe_get(seal_detection_result.get("result", {}), "details", {}).get("stamp", [])
            if stamp_list:
                seal_info = stamp_list[0]

    finally:
        # 清理所有临时文件
        try:
            # 删除转换后的图片文件
            for path in temp_image_paths:
                if os.path.exists(path):
                    os.remove(path)
            # 删除原始上传的文件
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as cleanup_error:
            # 如果清理失败，记录日志或处理
            print(f"清理临时文件时出错: {cleanup_error}")

    # 返回合并后的结果
    response = {
        "status_code": 200,
        "message": "success",
        "result": {
            "table": table_data if table_data else None,
            "seal": seal_info  # 如果印章识别失败，seal_info 为 None
        }
    }

    return response

async def recognize_seal(file_path: str):
    """调用印章检测接口并返回结果，支持PDF和图片文件"""
    try:
        # 确保文件存在
        if not os.path.exists(file_path):
            return {"error": f"文件路径不存在: {file_path}"}

        # 发送POST请求调用印章检测接口
        with open(file_path, 'rb') as file:
            files = {'image': file}  # 保持与客户端上传时的字段名一致
            response = requests.post(SEAL_RECOGNIZE_URL, files=files)

        # 解析印章检测的响应结果
        if response.status_code == 200:
            seal_data = response.json()  # 假设响应数据是JSON格式
            return seal_data
        else:
            return {"error": "印章检测接口响应失败", "status_code": response.status_code}

    except Exception as e:
        return {"error": f"印章检测请求失败: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=13006)
