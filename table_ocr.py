# table_ocr.py
import argparse
import os
import json
from bs4 import BeautifulSoup  # 用于解析HTML
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

class TableOCR:
    def __init__(self, model_type="yolox", output_dir="outputs"):
        self.lineless_engine = LinelessTableRecognition()
        self.wired_engine = WiredTableRecognition()
        self.table_cls = TableCls(model_type=model_type)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def perform_ocr(self, img_path):
        cls, elasp_cls = self.table_cls(img_path)
        if cls == 'wired':
            table_engine = self.wired_engine
        else:
            table_engine = self.lineless_engine

        # 执行表格识别
        html, elasp_engine, polygons, logic_points, ocr_res, dict = table_engine(img_path, version="v2", enhance_box_line=True, rotated_fix=True)
        print(f"Engine elapsed time: {elasp_engine} seconds")
        print("HTML Output:")
        print(html)
        print(dict)
        # 格式化HTML
        complete_html = format_html(html)
        html_path = os.path.join(self.output_dir, "table.html")
        with open(html_path, "w", encoding="utf-8") as file:
            file.write(complete_html)

        # 可视化表格识别框和逻辑行列信息
        rec_box_path = os.path.join(self.output_dir, "table_rec_box.jpg")
        plot_rec_box_with_logic_info(
            img_path, rec_box_path, logic_points, polygons
        )
        print(f"Recognition box image saved to: {rec_box_path}")

        # 可视化 OCR 识别框
        ocr_box_path = os.path.join(self.output_dir, "ocr_box.jpg")
        plot_rec_box(img_path, ocr_box_path, ocr_res)
        print(f"OCR box image saved to: {ocr_box_path}")

        # 解析HTML以提取单元格文本
        cells_text = self.extract_text_from_html(complete_html)

        # 构建JSON数据
        # json_data = self.build_json(logic_points, polygons, cells_text, dict)
        json_data = self.build_json(dict)

        # 保存JSON文件
        json_path = os.path.join(self.output_dir, "table.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        print(f"JSON output saved to: {json_path}")

        return json_path, elasp_cls

    def extract_text_from_html(self, html_content):
        """
        解析HTML内容，提取每个单元格的文本。
        默认HTML中的单元格按顺序排列，与logic_points和polygons对应。
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        cells = soup.find_all(['td', 'th'])  # 查找所有单元格
        cells_text = [cell.get_text(separator='\n', strip=True) for cell in cells]
        return cells_text

    """def build_json(self, logic_points, polygons, cells_text, dict):
        
        #根据logic_points、polygons和cells_text构建所需的JSON结构。
        
        #if not (len(logic_points) == len(polygons) == len(cells_text)):
         #   raise ValueError("逻辑点、多边形和文本数量不匹配。")

        json_cells = []
        for idx, (logic, poly, text) in enumerate(zip(logic_points, polygons, cells_text)):
            row_start, row_end, col_start, col_end = logic
            if len(poly) != 4:
                raise ValueError(f"多边形坐标长度不为4，实际长度为{len(poly)}。")
            x1, y1, x2, y2 = poly  # 假设多边形格式为 [x1, y1, x2, y2]

            # 转换为标准的Python float类型
            # position = [float(x1), float(y1), float(x2), float(y2),]
            position = [int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]

            cell_data = {
                "col_start": int(col_start),
                "col_end": int(col_end),
                "row_start": int(row_start),
                "row_end": int(row_end),
                "position": position,
                "text": text
            }
            json_cells.append(cell_data)
        return {"tables": json_cells}"""

    def build_json(self, ocr_result):
        """
        根据从process_ocr_result函数获得的OCR结果构建所需的JSON结构。
        """
        json_cells = []
        for entry in ocr_result:
            # 提取t_logic_box数据
            row_start, row_end, col_start, col_end = entry['t_logic_box']

            # 合并't_ocr_res'中所有的文本
            if entry['t_ocr_res']:
                text = " ".join([res[1] for res in entry['t_ocr_res']])
            else:
                text = ""

            # 提取位置
            x1, y1, x2, y2 = entry['t_box']
            # position 格式: [x1, y1, x2, y1, x2, y2, x1, y2]
            position = [int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]

            # 构建单元格数据
            cell_data = {
                "col_start": int(col_start),
                "col_end": int(col_end),
                "row_start": int(row_start),
                "row_end": int(row_end),
                "position": position,
                "text": text
            }
            json_cells.append(cell_data)

        return {"tables": json_cells}



def main():
    """
    主函数用于测试 TableOCR 类。
    在代码中直接定义待识别的图片路径、模型类型和输出目录。
    """
    # 定义参数
    image_path = "preprocessed_images/image1.jpg"  # 替换为您的图像文件路径
    model_type = "yolox"  # 可选: "yolox", 其他模型类型视情况而定
    output_dir = "outputs"  # 输出文件夹路径

    # 打印参数信息
    print("=== TableOCR 测试开始 ===")
    print(f"待识别的图像路径: {image_path}")
    print(f"使用的模型类型: {model_type}")
    print(f"输出目录: {output_dir}")
    print("==========================\n")

    # 检查输入图像文件是否存在
    if not os.path.isfile(image_path):
        print(f"错误: 图像文件 '{image_path}' 不存在。请检查路径是否正确。")
        return

    # 创建 TableOCR 实例
    table_ocr = TableOCR(model_type=model_type, output_dir=output_dir)

    # 执行 OCR
    try:
        json_path, elapsed_time = table_ocr.perform_ocr(image_path)
        print(f"\nOCR 处理完成，用时 {elapsed_time} 秒。")
        print(f"JSON 输出路径: {json_path}")
    except Exception as e:
        print(f"在执行 OCR 时发生错误: {e}")

if __name__ == "__main__":
    main()