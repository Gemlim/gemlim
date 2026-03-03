"""
多人AI体育教育测评系统 - 固定点位简化版
使用全局检测器，然后映射到固定点位
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from detecor import MultiPersonJumpRopeDetector
from position_mapper import PositionMapper

# 设置UTF-8输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class JumpRopeEvaluatorSimple:
    """跳绳测评器 - 固定点位简化版"""
    
    def __init__(self, model_path='yolo11l-pose.pt', calibration_file='position_calibration.json'):
        """初始化模型和点位映射器"""
        print("初始化YOLO模型...")
        self.model = YOLO(model_path)
        print("✅ 模型加载成功")
        
        # 加载点位映射器
        self.position_mapper = PositionMapper(num_positions=11)
        if not self.position_mapper.load_calibration(calibration_file):
            print(f"❌ 无法加载点位校准文件: {calibration_file}")
            sys.exit(1)
        
        print("ℹ️  检测模式：全局检测，映射到固定点位")
        print("ℹ️  点位模式：固定11个点位\n")
    
    def process_video(self, video_path):
        """处理单个视频文件"""
        print(f"\n处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return {}
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        video_duration = total_frames / fps
        print(f"   总帧数: {total_frames}, FPS: {fps:.1f}, 时长: {video_duration:.1f}秒")
        
        # 使用全局检测器（最多检测20人，然后映射到11个点位）
        global_detector = MultiPersonJumpRopeDetector(max_persons=20)
        
        # 记录每个tracker ID映射到的点位
        tracker_to_position = {}
        # 记录每个点位的所有tracker ID（用于累加）
        position_trackers = {pos_id: [] for pos_id in range(1, 12)}
        # 记录每个点位的最大跳绳次数（关键改进）
        position_max_jumps = {pos_id: 0 for pos_id in range(1, 12)}
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # 运行YOLO检测
            results = self.model(frame, verbose=False)
            
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
            
            # 将每个tracker映射到点位，并记录最大跳绳次数
            for tracker_id, tracker in global_detector.trackers.items():
                # 获取tracker的bbox
                if hasattr(tracker, 'last_bbox') and tracker.last_bbox is not None:
                    bbox = tracker.last_bbox
                    position_id = self.position_mapper.map_detection_to_position(bbox)
                    
                    if position_id is not None:
                        # 记录这个tracker属于哪个点位
                        if tracker_id not in tracker_to_position:
                            tracker_to_position[tracker_id] = position_id
                            position_trackers[position_id].append(tracker_id)
                        
                        # 更新该点位的最大跳绳次数
                        current_jumps = tracker.get_statistics()['jump_count']
                        position_max_jumps[position_id] = max(position_max_jumps[position_id], current_jumps)
            
            # 显示进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        
        # 统计每个点位的跳绳次数（使用最大值）
        position_stats = {}
        valid_positions = list(range(1, 6)) + list(range(7, 12))  # 1-5, 7-11
        
        for pos_id in valid_positions:
            # 使用历史最大值（防止检测丢失导致计数减少）
            position_stats[pos_id] = position_max_jumps[pos_id]
        
        total_jumps = sum(position_stats.values())
        print(f"✅ 处理完成: {len(valid_positions)}个点位 (1-5, 7-11), 总跳绳次数: {total_jumps}")
        
        # 打印每个点位的详细信息
        for pos_id in valid_positions:
            jump_count = position_stats[pos_id]
            tracker_count = len(position_trackers[pos_id])
            print(f"   点位 {pos_id}: {jump_count} 次 ({tracker_count}个tracker)")
        
        return position_stats
    
    def process_all_videos(self, input_dir, output_dir):
        """处理输入目录中的所有视频"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 查找所有视频文件
        video_files = sorted(input_path.glob('*.mp4'))
        
        if not video_files:
            print(f"❌ 在 {input_dir} 中没有找到视频文件")
            return
        
        print(f"\n找到 {len(video_files)} 个视频文件\n")
        
        all_results = []
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"处理视频 {i}/{len(video_files)}: {video_file.name}")
            print(f"{'='*60}")
            
            # 处理视频
            position_stats = self.process_video(str(video_file))
            
            # 提取视频编号
            video_number = ''.join(filter(str.isdigit, video_file.stem))
            if not video_number:
                video_number = str(i)
            
            # 生成CSV文件
            output_csv = output_path / f'result{video_number}.csv'
            
            print(f"\n{'='*60}")
            print(f"写入结果文件: result{video_number}.csv")
            print(f"{'='*60}")
            
            with open(output_csv, 'w', encoding='utf-8') as f:
                # 写入所有11个点位（点位6固定为0）
                for pos_id in range(1, 12):
                    if pos_id == 6:
                        jump_count = 0
                    else:
                        jump_count = position_stats.get(pos_id, 0)
                    
                    f.write(f"{pos_id},{jump_count}\n")
                    print(f"点位 {pos_id}: {jump_count} 次")
                    
                    all_results.append({
                        'location': pos_id,
                        'jumps': jump_count,
                        'filename': video_file.name
                    })
            
            print(f"✅ 结果已保存到: {output_csv}")
        
        # 显示汇总
        print(f"\n{'='*60}")
        print("汇总统计")
        print(f"{'='*60}")
        total_all = sum(r['jumps'] for r in all_results)
        print(f"  处理视频数: {len(video_files)}")
        print(f"  固定点位数: 11 (点位6固定为0)")
        print(f"  总跳绳次数: {total_all}")
        if len(all_results) > 0:
            total_without_pos6 = sum(r['jumps'] for r in all_results if r['location'] != 6)
            print(f"  平均每点位: {total_without_pos6/10/len(video_files):.1f} 次 (不含点位6)")


def main():
    """主函数"""
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        input_dir = './input_data'
        output_dir = './output_data'
    
    evaluator = JumpRopeEvaluatorSimple()
    evaluator.process_all_videos(input_dir, output_dir)
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
