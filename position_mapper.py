"""
点位位置映射器
用于在多个视频中绑定固定的11个点位位置
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PositionMapper:
    """点位位置映射器 - 将检测到的人映射到固定的11个点位"""
    
    def __init__(self, num_positions: int = 11):
        """
        初始化位置映射器
        
        Args:
            num_positions: 固定点位数量（默认11个）
        """
        self.num_positions = num_positions
        self.position_centers = []  # 每个点位的中心坐标 [(x, y), ...]
        self.is_calibrated = False
        
    def calibrate_from_first_frame(self, detections: List[Tuple], frame_width: int, frame_height: int):
        """
        从第一帧的检测结果校准点位位置
        
        Args:
            detections: 检测结果列表 [(keypoints, confidence, bbox), ...]
            frame_width: 视频宽度
            frame_height: 视频高度
        """
        if self.is_calibrated:
            return
        
        if len(detections) < self.num_positions:
            print(f"[WARNING] 第一帧只检测到 {len(detections)} 人，少于期望的 {self.num_positions} 个点位")
            # 继续处理，但点位数量会少于预期
        
        # 提取所有人的中心位置
        centers = []
        for detection in detections:
            keypoints, confidence, bbox = detection
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        # 按Y坐标排序，分成上下两排
        centers_sorted = sorted(centers, key=lambda c: c[1])
        
        # 如果有11个人，分成上排6人，下排5人
        if len(centers_sorted) >= 11:
            # 只取前11个
            centers_sorted = centers_sorted[:11]
            mid_idx = 6  # 上排6人
        else:
            # 如果少于11人，平均分配
            mid_idx = len(centers_sorted) // 2
        
        # 上排和下排
        top_row = sorted(centers_sorted[:mid_idx], key=lambda c: c[0])  # 按X排序
        bottom_row = sorted(centers_sorted[mid_idx:], key=lambda c: c[0])  # 按X排序
        
        # 点位编号：下排1-5（或更少），上排6-11（或更少）
        self.position_centers = bottom_row + top_row
        
        self.is_calibrated = True
        
        print(f"[INFO] 点位校准完成：检测到 {len(self.position_centers)} 个点位")
        print(f"      下排 {len(bottom_row)} 个点位 (ID 1-{len(bottom_row)})")
        print(f"      上排 {len(top_row)} 个点位 (ID {len(bottom_row)+1}-{len(self.position_centers)})")
    
    def map_detection_to_position(self, bbox: Tuple[float, float, float, float]) -> Optional[int]:
        """
        将检测框映射到最近的点位
        
        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            点位ID (1-based)，如果距离太远则返回None
        """
        if not self.is_calibrated:
            return None
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 找到最近的点位
        min_distance = float('inf')
        best_position = None
        
        for idx, (px, py) in enumerate(self.position_centers):
            distance = np.sqrt((center_x - px)**2 + (center_y - py)**2)
            if distance < min_distance:
                min_distance = distance
                best_position = idx + 1  # 1-based ID
        
        # 如果距离太远（超过150像素），认为不属于任何点位
        if min_distance > 150:
            return None
        
        return best_position
    
    def save_calibration(self, filepath: str):
        """保存校准数据到文件"""
        # 转换为Python原生类型
        position_centers_list = [[float(x), float(y)] for x, y in self.position_centers]
        
        data = {
            'num_positions': int(self.num_positions),
            'position_centers': position_centers_list,
            'is_calibrated': bool(self.is_calibrated)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] 点位校准数据已保存到: {filepath}")
    
    def load_calibration(self, filepath: str) -> bool:
        """从文件加载校准数据"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.num_positions = data['num_positions']
            self.position_centers = [tuple(c) for c in data['position_centers']]
            self.is_calibrated = data['is_calibrated']
            print(f"[INFO] 已加载点位校准数据: {len(self.position_centers)} 个点位")
            return True
        except FileNotFoundError:
            print(f"[WARNING] 校准文件不存在: {filepath}")
            return False
        except Exception as e:
            print(f"[ERROR] 加载校准文件失败: {e}")
            return False


def create_position_calibration(video_path: str, output_path: str = 'position_calibration.json'):
    """
    从第一个视频创建点位校准文件
    
    Args:
        video_path: 视频文件路径
        output_path: 输出校准文件路径
    """
    import cv2
    from ultralytics import YOLO
    
    print(f"从视频创建点位校准: {video_path}")
    
    # 加载模型
    model = YOLO('yolo11l-pose.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] 无法读取视频第一帧")
        return None
    
    # 运行检测
    print("检测第一帧...")
    results = model(frame, verbose=False)
    
    # 解析检测结果
    detections = []
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for i in range(len(keypoints)):
            kpts = keypoints[i, :, :2]
            conf = keypoints[i, :, 2]
            bbox = boxes[i]
            detections.append((kpts, conf, tuple(bbox)))
    
    print(f"检测到 {len(detections)} 人")
    
    # 创建映射器并校准
    mapper = PositionMapper(num_positions=11)
    mapper.calibrate_from_first_frame(detections, width, height)
    
    # 保存校准数据
    mapper.save_calibration(output_path)
    
    return mapper


if __name__ == "__main__":
    # 从第一个视频创建校准文件
    create_position_calibration('input_data/1.mp4', 'position_calibration.json')
