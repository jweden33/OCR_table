import os

from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from rapidocr_onnxruntime import RapidOCR
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

lineless_engine = LinelessTableRecognition()
wired_engine = WiredTableRecognition()
# 默认小yolo模型(0.1s)，可切换为精度更高yolox(0.25s),更快的qanything(0.07s)模型
table_cls = TableCls(model_type="yolox")  # TableCls(model_type="yolox"),TableCls(model_type="q")
img_path = f'付款申请单-3.jpg'

cls, elasp = table_cls(img_path)
if cls == 'wired':
    table_engine = wired_engine
else:
    table_engine = lineless_engine

html, elasp, polygons, logic_points, ocr_res = table_engine(img_path)
#print(f"elasp: {elasp}")
print(html)
print(polygons)
print(logic_points)
print(ocr_res)
# 使用其他ocr模型
# ocr_engine =RapidOCR(det_model_dir="xxx/det_server_infer.onnx",rec_model_dir="xxx/rec_server_infer.onnx")
# ocr_res, _ = ocr_engine(img_path)
# html, elasp, polygons, logic_points, ocr_res = table_engine(img_path, ocr_result=ocr_res)
output_dir = f'outputs'
complete_html = format_html(html)
os.makedirs(os.path.dirname(f"{output_dir}/table.html"), exist_ok=True)
with open(f"{output_dir}/table.html", "w", encoding="utf-8") as file:
    file.write(complete_html)
# # 可视化表格识别框 + 逻辑行列信息
plot_rec_box_with_logic_info(
   img_path, f"{output_dir}/table_rec_box.jpg", logic_points, polygons
 )
# # 可视化 ocr 识别框
plot_rec_box(img_path, f"{output_dir}/ocr_box.jpg", ocr_res)