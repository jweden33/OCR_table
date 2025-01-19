# orientation_correction.py

import os
import cv2
from rapid_table_det.inference import TableDetector
from rapid_table_det.utils.visuallize import img_loader, visuallize, extract_table_img

class ImageOrientationCorrector:
    def __init__(self, output_dir="rapid_table_det/outputs"):
        self.table_det = TableDetector()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def correct_orientation(self, img_path):
        result, elapse = self.table_det(img_path)
        obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
        print(
            f"obj_det_elapse: {obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}"
        )

        img = img_loader(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file_name_with_ext = os.path.basename(img_path)
        file_name, _ = os.path.splitext(file_name_with_ext)
        extract_img = img.copy()

        corrected_image_paths = []
        for i, res in enumerate(result):
            box = res["box"]
            lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
            # 可视化识别框和方向
            visuallize(img, box, lt, rt, rb, lb)
            # 提取并矫正表格图片
            wrapped_img = extract_table_img(extract_img.copy(), lt, rt, rb, lb)
            corrected_image_path = os.path.join(self.output_dir, f"{file_name}-extract-{i}.jpg")
            cv2.imwrite(corrected_image_path, wrapped_img)
            corrected_image_paths.append(corrected_image_path)

        # 保存可视化结果
        visualize_path = os.path.join(self.output_dir, f"{file_name}-visualize.jpg")
        cv2.imwrite(visualize_path, img)

        return corrected_image_paths, elapse
