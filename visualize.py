"""
生成带检测框的可视化视频 - 使用固定点位系统
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from detecor import MultiPersonJumpRopeDetector
from position_mapper import PositionMapper

def visualize_video(video_path, output_path, calibration_file='position_calibration.json'):
    """
    生成带检测框和跳绳计数的可视化视频
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        calibration_file: 点位校准文件
    """
    print(f"处理视频: {video_path}")
    
    # 加载模型
    model = YOLO('yolo11l-pose.pt')
    
    # 加载点位映射器
    position_mapper = PositionMapper(num_positions=11)
    if not position_mapper.load_calibration(calibration_file):
        print(f"❌ 无法加载点位校准文件")
        return
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 无法创建输出视频")
        cap.release()
        return
    
    # 创建全局检测器
    global_detector = MultiPersonJumpRopeDetector(max_persons=20)
    
    # 记录tracker到点位的映射
    tracker_to_position = {}
    position_trackers = {pos_id: [] for pos_id in range(1, 12)}
    position_max_jumps = {pos_id: 0 for pos_id in range(1, 12)}
    
    # 颜色定义（11种颜色）
    colors = [
        (255, 0, 0),    # 1: 蓝色
        (0, 255, 0),    # 2: 绿色
        (0, 0, 255),    # 3: 红色
        (255, 255, 0),  # 4: 青色
        (255, 0, 255),  # 5: 品红
        (0, 255, 255),  # 6: 黄色
        (128, 0, 0),    # 7: 深蓝
        (0, 128, 0),    # 8: 深绿
        (0, 0, 128),    # 9: 深红
        (128, 128, 0),  # 10: 深青
        (128, 0, 128),  # 11: 深品红
    ]
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # 运行YOLO检测
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
        
        # 更新全局检测器
        global_detector.update(detections, current_time)
        
        # 将tracker映射到点位并更新最大跳绳次数
        for tracker_id, tracker in global_detector.trackers.items():
            if hasattr(tracker, 'last_bbox') and tracker.last_bbox is not None:
                bbox = tracker.last_bbox
                position_id = position_mapper.map_detection_to_position(bbox)
                
                if position_id is not None:
                    # 记录这个tracker属于哪个点位
                    if tracker_id not in tracker_to_position:
                        tracker_to_position[tracker_id] = position_id
                        position_trackers[position_id].append(tracker_id)
                    
                    # 更新该点位的最大跳绳次数
                    current_jumps = tracker.get_statistics()['jump_count']
                    position_max_jumps[position_id] = max(position_max_jumps[position_id], current_jumps)
                    
                    # 绘制检测框
                    x1, y1, x2, y2 = map(int, bbox)
                    color = colors[(position_id - 1) % len(colors)]
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # 显示点位ID和跳绳次数（使用最大值）
                    max_jumps = position_max_jumps[position_id]
                    label = f"P{position_id}: {max_jumps}"
                    
                    # 绘制标签背景
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, 
                                 (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0] + 10, y1),
                                 color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 在左上角显示当前帧信息
        info_text = f"Frame: {frame_count}/{total_frames} | Time: {current_time:.1f}s"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 在右上角显示点位统计
        y_offset = 30
        for pos_id in range(1, 12):
            if position_max_jumps[pos_id] > 0:
                stat_text = f"P{pos_id}: {position_max_jumps[pos_id]}"
                color = colors[(pos_id - 1) % len(colors)]
                cv2.putText(frame, stat_text, (width - 120, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        # 写入帧
        out.write(frame)
        
        # 显示进度
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    out.release()
    
    print(f"✅ 可视化视频已保存: {output_path}")
    
    # 显示最终统计
    print("\n最终点位统计:")
    valid_positions = list(range(1, 6)) + list(range(7, 12))
    for pos_id in valid_positions:
        if position_max_jumps[pos_id] > 0:
            print(f"   点位 {pos_id}: {position_max_jumps[pos_id]} 次")


def main():
    """主函数 - 处理所有视频"""
    import sys
    
    input_dir = Path('./input_data')
    output_dir = Path('./output_data/visualized')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 查找所有视频
    video_files = sorted(input_dir.glob('*.mp4'))
    
    if not video_files:
        print("❌ 没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(video_files)}] {video_file.name}")
        print(f"{'='*60}")
        
        output_file = output_dir / f'{video_file.stem}_detected.avi'
        visualize_video(str(video_file), str(output_file))
    
    print(f"\n{'='*60}")
    print(f"✅ 所有视频处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
