import os
import tempfile
import logging
import requests
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import shutil

from orientation_correction import ImageOrientationCorrector
from table_ocr import TableOCR

from pdf2image import convert_from_path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)

# 配置上传文件的限制（最大 50MB）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pdf'}

# 定义预处理图像的保存目录
PREPROCESSED_DIR = 'preprocessed_images'
os.makedirs(PREPROCESSED_DIR, exist_ok=True)  # 确保目录存在

# 检查文件扩展名是否在允许的扩展名列表中
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 安全获取字典中的值
def safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default

def resize_image(input_path, output_path, max_width=1200):
    """
    调整图像尺寸，使其宽度不超过max_width，同时保持纵横比。
    参数:
    - input_path: 输入图像的路径
    - output_path: 输出图像的路径
    - max_width: 图像的最大宽度（默认为1500像素）
    """
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            if width > max_width:
                ratio = max_width / float(width)
                new_size = (max_width, int(height * ratio))
                # 使用 Image.LANCZOS 代替 Image.ANTIALIAS
                img = img.resize(new_size, Image.LANCZOS)
                img.save(output_path)
                logger.info(f"已调整尺寸: {input_path} -> {output_path} 尺寸: {new_size}")
                return output_path
            else:
                logger.info(f"无需调整尺寸: {input_path} 保持原尺寸: {img.size}")
                return input_path  # 如果不需要调整，返回原路径
    except Exception as e:
        logger.error(f"调整图像尺寸时出错: {input_path} 错误信息: {e}")
        return input_path  # 出错时返回原路径

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({
            "code": 40101,
            "message": "No image part in the request",
            "result": {}
        }), 40101

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "code": 40102,
            "message": "No selected file",
            "result": {}
        }), 40102

    if file and allowed_file(file.filename):
        original_filename = file.filename
        filename = secure_filename(original_filename)

        # 生成唯一且安全的文件名
        unique_id = uuid.uuid4().hex
        file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        filename = f"{unique_id}.{file_extension}" if file_extension else unique_id

        try:
            # 创建一个临时目录来存储上传的文件和处理结果
            with tempfile.TemporaryDirectory() as temp_dir:
                input_file_path = os.path.join(temp_dir, filename)
                file.save(input_file_path)
                logger.info(f"文件已保存到: {input_file_path}")

                # 初始化用于收集所有表格和印章识别结果的列表
                all_tables = []
                all_seals = []

                if file_extension == 'pdf':
                    # 调用印章识别接口（处理整个 PDF 文件）
                    seal_recognition_result = call_seal_recognition_api(input_file_path)

                    # 提取 seal_info，只取 stamp_list[0]
                    if seal_recognition_result and 'result' in seal_recognition_result:
                        stamp_list = safe_get(seal_recognition_result.get('result', {}), 'details', {}).get('stamp', [])
                        if stamp_list:
                            seal_info = stamp_list[0]
                        else:
                            seal_info = {"message": "No stamps detected"}
                    else:
                        seal_info = {"error": "Seal recognition failed"}

                    all_seals = seal_info  # 对于 PDF 文件，seal 为单个对象

                    # 将 PDF 转换为图像
                    try:
                        logger.info("正在将 PDF 转换为图像...")
                        images = convert_from_path(input_file_path)
                        logger.info(f"PDF 已转换为 {len(images)} 张图像。")
                    except Exception as e:
                        logger.error(f"转换 PDF 为图像时出错: {e}")
                        return jsonify({
                            "code": 40103,
                            "message": f"Error converting PDF to images: {str(e)}",
                            "result": {}
                        }), 40103

                    for page_number, image in enumerate(images, start=1):
                        # 将每一页保存为图像文件
                        page_image_path = os.path.join(temp_dir, f"page_{page_number}.png")
                        image.save(page_image_path, 'PNG')
                        logger.info(f"已保存第 {page_number} 页为: {page_image_path}")

                        # 调整每页图像的尺寸
                        resized_page_image_path = os.path.join(temp_dir, f"resized_page_{page_number}.png")
                        resized_page_path = resize_image(page_image_path, resized_page_image_path, max_width=1200)
                        final_page_path = resized_page_path if resized_page_path != page_image_path else page_image_path

                        # 初始化方向矫正器，每页使用独立的输出目录
                        orientation_corrector = ImageOrientationCorrector(output_dir=os.path.join(temp_dir, "outputs", f"page_{page_number}"))
                        corrected_images, orientation_elapse = orientation_corrector.correct_orientation(final_page_path)
                        logger.info(f"第 {page_number} 页的方向矫正完成，用时 {orientation_elapse} 秒。")
                        logger.info(f"第 {page_number} 页的矫正后图像: {corrected_images}")

                        if not corrected_images:
                            logger.warning(f"第 {page_number} 页的方向矫正失败，跳过。")
                            continue

                        # 初始化 OCR 表格识别器
                        table_ocr = TableOCR(model_type="yolox", output_dir=os.path.join(temp_dir, "ocr_outputs", f"page_{page_number}"))

                        # 对每个矫正后的图像执行 OCR 识别
                        for corrected_image in corrected_images:
                            json_path, ocr_elapse = table_ocr.perform_ocr(corrected_image)
                            logger.info(f"OCR 完成，用时 {ocr_elapse} 秒。")
                            logger.info(f"OCR 输出已保存到: {json_path}")

                            if not os.path.isfile(json_path):
                                logger.error(f"OCR 失败，图像: {corrected_image}")
                                continue

                            # 读取 JSON 文件内容
                            with open(json_path, 'r', encoding='utf-8') as json_file:
                                ocr_data = json.load(json_file)

                            if "tables" in ocr_data:
                                all_tables.extend(ocr_data["tables"])

                else:
                    # 处理图像文件

                    # 调整上传的图片尺寸
                    resized_input_path = os.path.join(temp_dir, f"resized_{filename}")
                    resized_path = resize_image(input_file_path, resized_input_path, max_width=1200)

                    # 如果调整后的路径与原路径不同，使用调整后的路径
                    final_input_path = resized_path if resized_path != input_file_path else input_file_path

                    # 调用印章识别接口（在方向矫正之前）
                    seal_recognition_result = call_seal_recognition_api(final_input_path)

                    # 提取 seal_info，只取 stamp_list[0]
                    if seal_recognition_result and 'result' in seal_recognition_result:
                        stamp_list = safe_get(seal_recognition_result.get('result', {}), 'details', {}).get('stamp', [])
                        if stamp_list:
                            seal_info = stamp_list[0]
                        else:
                            seal_info = {"message": "No stamps detected"}
                    else:
                        seal_info = {"error": "Seal recognition failed"}

                    all_seals = seal_info  # 对于图像文件，seal 为单个对象

                    # 初始化方向矫正器
                    orientation_corrector = ImageOrientationCorrector(output_dir=os.path.join(temp_dir, "outputs"))
                    corrected_images, orientation_elapse = orientation_corrector.correct_orientation(final_input_path)
                    logger.info(f"方向矫正完成，用时 {orientation_elapse} 秒。")
                    logger.info(f"矫正后图像: {corrected_images}")

                    if not corrected_images:
                        return jsonify({
                            "code": 430,
                            "message": "Orientation correction failed",
                            "result": {}
                        }), 200
                    # 将每个矫正后的图像保存到 preprocessed_images 目录
                    for corrected_image in corrected_images:
                        # 生成唯一文件名以避免冲突
                        preprocessed_filename = f"image_{unique_id}_{os.path.basename(corrected_image)}"
                        permanent_corrected_path = os.path.join(PREPROCESSED_DIR, preprocessed_filename)
                        shutil.copy(corrected_image, permanent_corrected_path)
                        logger.info(f"预处理后的图像已保存到: {permanent_corrected_path}")

                    # 初始化 OCR 表格识别器
                    table_ocr = TableOCR(model_type="yolox", output_dir=os.path.join(temp_dir, "ocr_outputs"))

                    # 对每个矫正后的图像执行 OCR 识别
                    for corrected_image in corrected_images:
                        json_path, ocr_elapse = table_ocr.perform_ocr(corrected_image)
                        logger.info(f"OCR 完成，用时 {ocr_elapse} 秒。")
                        logger.info(f"OCR 输出已保存到: {json_path}")
                        if not os.path.isfile(json_path):
                            logger.error(f"OCR 失败，图像: {corrected_image}")
                            continue

                        # 读取 JSON 文件内容
                        with open(json_path, 'r', encoding='utf-8') as json_file:
                            ocr_data = json.load(json_file)

                        if "tables" in ocr_data:
                            all_tables.extend(ocr_data["tables"])

                # 构建响应的 JSON 结构
                response = {
                    "code": 200,
                    "message": "success",
                    "result": {
                        "table": {
                            "details": [],
                            "result": {
                                "tables": all_tables
                            }
                        },
                        "seal": all_seals  # 'seal' 为列表或单个对象
                    }
                }

                return jsonify(response), 200

        except Exception as e:
            logger.error(f"处理图像时出错: {e}")
            return jsonify({
                "code": 500,
                "message": f"Internal server error: {str(e)}",
                "result": {}
            }), 500
    else:
        return jsonify({
            "code": 40104,
            "message": "Unsupported file type",
            "result": {}
        }), 40103

def call_seal_recognition_api(file_path):
    """
    调用印章识别检测接口，并返回 seal_data。
    对于 PDF 文件，直接发送 PDF；对于图像文件，发送图像。
    """
    seal_api_url = 'http://h1337.iis.pub:24221/seal/recognize_seal'
    try:
        with open(file_path, 'rb') as file:
            files = {'image': file}
            logger.info(f"发送文件到印章识别 API: {seal_api_url}")
            response = requests.post(seal_api_url, files=files, timeout=30)  # 设置超时时间为30秒

        if response.status_code == 200:
            seal_data = response.json()
            logger.info(f"印章识别 API 响应: {seal_data}")
            return seal_data  # 假设 seal_data 是一个字典
        else:
            logger.error(f"印章识别 API 返回状态码 {response.status_code}")
            return {"error": f"Seal recognition API returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"调用印章识别 API 时出错: {e}")
        return {"error": f"Error calling seal recognition API: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"解析印章识别 API 响应时出错: {e}")
        return {"error": "Invalid JSON response from seal recognition API"}

if __name__ == '__main__':
    # 运行 Flask 应用，监听所有可用 IP，端口号 13006
    app.run(host='0.0.0.0', port=13006)
