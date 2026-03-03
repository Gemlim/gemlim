"""
Jump Rope Detection GUI - English Version
跳绳检测图形界面 - 英文版本
"""
# 计算机视觉库，用于视频处理和图像操作
import cv2
# 时间处理模块，用于计时和延迟
import time
# Tkinter GUI库，用于创建图形界面
import tkinter as tk
# 多线程处理模块，用于并行处理
from threading import Thread, Event
# 队列模块，用于线程间通信
from queue import Queue, Empty
# Tkinter对话框模块，用于文件选择和消息提示
from tkinter import filedialog, messagebox
# PIL图像处理库，用于图像格式转换和显示
from PIL import Image, ImageTk
# YOLO目标检测模型库
from ultralytics import YOLO
# PyTorch 用于检测 GPU 支持
import torch
# 自定义跳绳检测器模块
from detecor import MultiPersonJumpRopeDetector
# 自定义位置映射器模块
from position_mapper import PositionMapper
# 路径处理模块，用于文件和目录操作
from pathlib import Path


class JumpRopeGUI:
    """跳绳检测图形界面主类"""
    def __init__(self, root):
        self.root = root
        self.root.title("Jump Rope Detection GUI")
        self.root.geometry("1600x900")
        self.root.configure(bg='#f5f5f5')
        
        # 模型和检测器相关变量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 推理设备
        self.model = None  # YOLO模型
        self.detector = None  # 跳绳检测器
        self.cap = None  # 视频捕获对象
        self.is_playing = False  # 是否正在播放
        self.is_camera = False  # 是否使用摄像头
        self.camera_loop_job = None  # 摄像头循环任务
        self.camera_index = 0  # 摄像头索引
        self.frame_count = 0  # 当前帧数
        self.total_frames = 0  # 总帧数
        self.fps = 30  # 帧率
        self.current_frame = None  # 当前帧
        self.current_video_path = None  # 当前视频路径
        self.result_written = False  # 结果是否已写入
        self.output_dir = Path('output_data')  # 输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 多线程处理相关变量
        self.frame_queue = None  # 帧队列
        self.result_queue = None  # 结果队列
        self.reader_thread = None  # 读取线程
        self.processor_thread = None  # 处理线程
        self.display_job = None  # 显示任务
        self.pipeline_stop = None  # 管道停止标志
        self.streaming_mode = None  # 流媒体模式
        
        # 不同人员显示颜色配置
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]
        
        # 初始化界面和组件
        self.create_ui()
        self.load_model()
        self.init_position_mapper()
        self.reset_position_stats()
        
    def create_ui(self):
        """创建用户界面"""
        # 顶部控制栏
        top_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        top_frame.pack_propagate(False)
        
        # 打开视频按钮
        tk.Button(top_frame, text="Open Video", command=self.load_video,
             font=('Arial', 16, 'bold'), bg='#27ae60', fg='white',
             padx=28, pady=14, relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=10, pady=10)
        
        # 播放/暂停按钮
        self.play_btn = tk.Button(top_frame, text="Play", command=self.toggle_play,
                      font=('Arial', 16, 'bold'), bg='#3498db', fg='white',
                      padx=28, pady=14, state=tk.DISABLED, relief=tk.FLAT, cursor='hand2')
        self.play_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # 下一帧按钮
        tk.Button(top_frame, text="Next Frame", command=self.next_frame,
             font=('Arial', 14), bg='#34495e', fg='white',
             padx=18, pady=13, relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=10)
        
        # 重置按钮
        tk.Button(top_frame, text="Reset", command=self.reset,
             font=('Arial', 14), bg='#e74c3c', fg='white',
             padx=22, pady=13, relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=10)

        # 导出结果按钮
        self.export_btn = tk.Button(top_frame, text="Export Results", command=self.manual_export,
                font=('Arial', 14, 'bold'), bg='#8e44ad', fg='white',
                padx=24, pady=13, relief=tk.FLAT, cursor='hand2', state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5, pady=10)

        # 摄像头按钮
        self.camera_btn = tk.Button(top_frame, text="Start Camera", command=self.toggle_camera,
                        font=('Arial', 14, 'bold'), bg='#1abc9c', fg='white',
                        padx=24, pady=13, relief=tk.FLAT, cursor='hand2')
        self.camera_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # 状态标签
        self.status_label = tk.Label(top_frame, text="Ready", font=('Arial', 15, 'bold'),
                                     bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack(side=tk.LEFT, padx=25)
        
        # 主界面区域
        main_frame = tk.Frame(self.root, bg='#f5f5f5')
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # 左侧：视频显示区域
        video_frame = tk.Frame(main_frame, bg='black', relief=tk.SUNKEN, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(video_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：统计信息面板
        info_frame = tk.Frame(main_frame, bg='white', width=380, relief=tk.RAISED, bd=2)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(15, 0))
        info_frame.pack_propagate(False)
        
        # 标题栏
        title_frame = tk.Frame(info_frame, bg='#34495e', height=60)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Detection Statistics", 
            font=('Arial', 22, 'bold'), bg='#34495e', fg='white').pack(pady=15)
        
        # 统计文本区域
        text_frame = tk.Frame(info_frame, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text = tk.Text(text_frame, font=('Courier New', 13), bg='#fafafa',
                                fg='#2c3e50', yscrollcommand=scrollbar.set, 
                                relief=tk.FLAT, padx=15, pady=10, wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)
        
    def init_position_mapper(self):
        """初始化位置映射器"""
        self.position_mapper = PositionMapper(num_positions=11)
        self.calibration_file = 'position_calibration.json'
        self.has_calibration = self.position_mapper.load_calibration(self.calibration_file)
        if not self.has_calibration:
            self.status_label.config(text="Calibration file missing", fg='#e74c3c')

    def reset_position_stats(self):
        """重置位置统计信息"""
        self.tracker_to_position = {}  # 跟踪器到位置的映射
        self.position_trackers = {pos_id: set() for pos_id in range(1, 12)}  # 位置跟踪器
        self.position_max_jumps = {pos_id: 0 for pos_id in range(1, 12)}  # 每个位置的最大跳绳次数
        self.valid_positions = list(range(1, 6)) + list(range(7, 12))  # 有效位置（排除6号位置）

    def manual_export(self):
        self.save_results(auto=False)

    def start_pipeline(self, mode):
        """启动处理管道（多线程）"""
        self.stop_pipeline()  # 先停止现有的管道
        self.streaming_mode = mode  # 设置流媒体模式
        self.pipeline_stop = Event()  # 创建停止事件
        self.frame_queue = Queue(maxsize=5)  # 帧队列（最多缓存5帧）
        self.result_queue = Queue(maxsize=3)  # 结果队列（最多缓存3个结果）
        
        # 创建并启动读取和处理线程
        self.reader_thread = Thread(target=self.frame_reader, daemon=True)
        self.processor_thread = Thread(target=self.processor_worker, daemon=True)
        self.reader_thread.start()
        self.processor_thread.start()
        self.schedule_display()  # 开始调度显示

    def stop_pipeline(self):
        """停止处理管道"""
        if self.pipeline_stop is not None:
            self.pipeline_stop.set()  # 设置停止标志
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1)  # 等待读取线程结束
        if self.processor_thread is not None:
            self.processor_thread.join(timeout=1)  # 等待处理线程结束
        if self.display_job is not None:
            self.root.after_cancel(self.display_job)  # 取消显示任务
            self.display_job = None
        # 重置所有管道相关变量
        self.reader_thread = None
        self.processor_thread = None
        self.frame_queue = None
        self.result_queue = None
        self.pipeline_stop = None
        self.streaming_mode = None

    def frame_reader(self):
        """帧读取线程：从视频或摄像头读取帧"""
        while self.pipeline_stop and not self.pipeline_stop.is_set():
            if not self.cap:
                break
            ret, frame = self.cap.read()
            if not ret:
                if self.streaming_mode == 'video':
                    # 视频播放结束
                    self.pipeline_stop.set()
                    self.root.after(0, self.on_stream_end)
                    break
                else:
                    # 摄像头信号丢失
                    self.root.after(0, lambda: self.status_label.config(text="Camera feed lost", fg='#e74c3c'))
                    time.sleep(0.05)
                    continue
            
            # 更新帧计数
            if self.streaming_mode == 'video':
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                self.frame_count += 1
                
            # 将帧放入队列供处理线程使用
            try:
                self.frame_queue.put(frame, timeout=0.1)
            except Exception:
                continue
        
    def processor_worker(self):
        """处理工作线程：分析帧并检测跳绳"""
        while self.pipeline_stop and not self.pipeline_stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)  # 从队列获取帧
            except Empty:
                continue
            
            self.current_frame = frame  # 保存当前帧
            processed, results = self.analyze_frame(frame)  # 分析帧
            if processed is None:
                continue
                
            # 将处理结果放入队列
            try:
                self.result_queue.put((processed, results), timeout=0.1)
            except Exception:
                continue

    def schedule_display(self):
        """调度显示：定期从结果队列获取并显示处理结果"""
        if self.pipeline_stop is None or (self.pipeline_stop and self.pipeline_stop.is_set()):
            return
        try:
            frame, results = self.result_queue.get_nowait()  # 非阻塞获取结果
            self.render_frame(frame, results)  # 渲染帧
        except Empty:
            pass
        # 每10毫秒调度下一次显示
        self.display_job = self.root.after(10, self.schedule_display)

    def on_stream_end(self):
        """流结束处理：视频播放完成时的回调"""
        self.stop_pipeline()  # 停止管道
        self.is_playing = False  # 设置播放状态为停止
        if not self.is_camera:
            self.play_btn.config(text="Play", bg='#3498db')  # 重置播放按钮
        self.status_label.config(text="Playback finished", fg='#95a5a6')  # 更新状态
        self.save_results(auto=True)  # 自动保存结果

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.is_camera:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """启动摄像头"""
        try:
            if self.cap:
                self.cap.release()
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = None
                messagebox.showerror("Error", f"Cannot open camera index {self.camera_index}")
                return
            try:
                # 设置摄像头缓冲区大小
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
                
            # 初始化摄像头相关状态
            self.is_camera = True
            self.is_playing = False
            self.frame_count = 0
            self.total_frames = 0
            self.detector = MultiPersonJumpRopeDetector(max_persons=20)
            self.reset_position_stats()
            self.current_video_path = None
            self.result_written = False
            
            # 更新UI状态
            self.play_btn.config(state=tk.DISABLED)
            self.camera_btn.config(text="Stop Camera", bg='#c0392b')
            self.status_label.config(text="Camera streaming...", fg='#27ae60')
            
            # 启动处理管道
            self.start_pipeline(mode='camera')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        """停止摄像头"""
        self.is_camera = False
        self.stop_pipeline()
        if self.cap:
            self.cap.release()
            self.cap = None
        # 恢复UI状态
        self.camera_btn.config(text="Start Camera", bg='#1abc9c')
        self.play_btn.config(state=tk.NORMAL if self.current_frame is not None else tk.DISABLED)
        self.status_label.config(text="Camera stopped", fg='#e67e22')

    def load_model(self):
        """加载YOLO模型"""
        try:
            self.status_label.config(text="Loading model...", fg='#f39c12')
            self.root.update()
            self.model = YOLO('yolo11l-pose.pt')
            self.model.to(self.device)
            device_name = 'GPU' if self.device == 'cuda' else 'CPU'
            self.status_label.config(text=f"Model loaded on {device_name}", fg='#27ae60')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            
    def load_video(self):
        """加载视频文件"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            initialdir="/root/jump/input_data",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            if self.is_camera:
                self.stop_camera()
                
            # 打开视频文件
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open video file")
                return
            
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.frame_count = 0
            
            # 初始化检测器
            self.detector = MultiPersonJumpRopeDetector(max_persons=20)
            self.reset_position_stats()
            self.current_video_path = Path(file_path)
            self.result_written = False
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取第一帧进行分析
            self.read_frame()
            frame_vis, results = self.analyze_frame(self.current_frame)
            self.render_frame(frame_vis, results)
            
            # 更新UI状态
            self.play_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            video_name = Path(file_path).name
            self.status_label.config(text=f"Loaded: {video_name} ({self.total_frames} frames)", fg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            
    def read_frame(self):
        """读取下一帧"""
        if not self.cap:
            return False
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return True
        return False
        
    def analyze_frame(self, frame):
        """分析帧：检测人体姿态和跳绳动作"""
        if frame is None or self.model is None:
            return frame, []
        work = frame.copy()
        video_time = self.frame_count / self.fps if self.fps else 0.0

        # 使用YOLO进行全身姿态检测
        results = self.model(work, device=self.device, verbose=False)

        # 提取检测结果
        detections = []
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i in range(len(keypoints)):
                kpts = keypoints[i, :, :2]  # 关键点坐标
                conf = keypoints[i, :, 2]   # 关键点置信度
                bbox = boxes[i]            # 边界框
                detections.append((kpts, conf, tuple(bbox)))

        # 更新跳绳检测器
        person_results = self.detector.update(detections, video_time)
        self.update_position_mapping()

        # 绘制结果到帧上
        work = self.draw_results(work, person_results)
        return work, person_results

    def update_position_mapping(self):
        """更新位置映射：将检测到的人员映射到指定位置"""
        if not self.position_mapper.is_calibrated:
            return  # 如果没有校准数据，直接返回
            
        for tracker_id, tracker in self.detector.trackers.items():
            bbox = getattr(tracker, 'last_bbox', None)
            if bbox is None:
                continue

            # 如果跟踪器已有位置映射，使用现有映射
            if tracker_id in self.tracker_to_position:
                position_id = self.tracker_to_position[tracker_id]
            else:
                # 否则尝试映射到新位置
                position_id = self.position_mapper.map_detection_to_position(bbox)
                if position_id is None:
                    continue
                self.tracker_to_position[tracker_id] = position_id
                self.position_trackers[position_id].add(tracker_id)

            # 跳过6号位置（通常为中心位置或无效位置）
            if position_id == 6:
                continue

            # 更新该位置的最大跳绳次数
            jump_count = tracker.get_statistics()['jump_count']
            self.position_max_jumps[position_id] = max(
                self.position_max_jumps[position_id], jump_count
            )

    def extract_video_identifier(self):
        """提取视频标识符：从视频文件名中提取数字或时间戳"""
        if self.current_video_path is None:
            return time.strftime('%Y%m%d%H%M%S')
        video_stem = self.current_video_path.stem
        digits = ''.join(ch for ch in video_stem if ch.isdigit())
        return digits if digits else video_stem

    def compute_position_stats(self):
        """计算位置统计：严格依赖点位映射"""
        stats = {pos_id: 0 for pos_id in range(1, 12)}
        stats[6] = 0

        if self.detector is None or not self.position_mapper.is_calibrated:
            return stats, False

        mapped_any = False
        for tracker_id, tracker in self.detector.trackers.items():
            bbox = getattr(tracker, 'last_bbox', None)
            if bbox is None:
                continue

            position_id = self.tracker_to_position.get(tracker_id)
            if position_id is None:
                position_id = self.position_mapper.map_detection_to_position(bbox)
                if position_id is None:
                    continue
                self.tracker_to_position[tracker_id] = position_id

            if position_id == 6 or position_id not in stats:
                continue

            jump_count = tracker.get_statistics().get('jump_count', 0)
            stats[position_id] = max(stats[position_id], jump_count)
            mapped_any = True

        return stats, mapped_any

    def save_results(self, auto=False):
        """保存结果到CSV文件"""
        if auto and self.result_written:
            return  # 如果已经自动保存过，不再重复保存
            
        # 检查是否有校准数据
        if not self.position_mapper.is_calibrated:
            if not auto:
                messagebox.showwarning("Export Failed", "Calibration data missing. Please provide position calibration before exporting.")
            self.status_label.config(text="Cannot export without calibration", fg='#e74c3c')
            return
            
        # 检查是否加载了视频
        if self.current_video_path is None:
            if not auto:
                messagebox.showwarning("Export Failed", "No video loaded. Please open a video first.")
            return

        # 计算位置统计
        position_stats, mapped_any = self.compute_position_stats()
        if not mapped_any:
            warn_msg = "No detections matched the calibrated positions. Please recalibrate before exporting."
            if not auto:
                messagebox.showwarning("Export Failed", warn_msg)
            self.status_label.config(text="Export blocked – recalibration required", fg='#e74c3c')
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        video_id = self.extract_video_identifier()
        output_csv = self.output_dir / f"result{video_id}.csv"

        # 写入CSV文件
        with open(output_csv, 'w', encoding='utf-8') as f:
            for pos_id in range(1, 12):
                jump_count = position_stats.get(pos_id, 0)
                f.write(f"{pos_id},{jump_count}\n")

        self.result_written = True
        if auto:
            self.status_label.config(text=f"Auto-saved: {output_csv.name}", fg='#27ae60')
        else:
            messagebox.showinfo("Export Complete", f"Results saved to {output_csv}")
            self.status_label.config(text=f"Exported: {output_csv.name}", fg='#27ae60')
        
    def draw_results(self, frame, results):
        """在帧上绘制检测结果"""
        for result in results:
            pid = result['person_id']  # 人员ID
            bbox = result['bbox']      # 边界框
            color = self.colors[(pid - 1) % len(self.colors)]  # 分配颜色
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # 创建标签：显示ID和跳绳次数
            label = f"#{pid}: {result['jump_count']} jumps"
            if result['is_timing']:
                label += f" [{result['elapsed_time']:.1f}s]"
            else:
                label += " (PAUSED)"
            
            # 绘制标签背景
            font_scale = 1.3
            thickness = 3
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1- (lh + 18)), (x1+lw+14, y1), color, -1)
            cv2.putText(frame, label, (x1+6, y1-6),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
            
            # 显示违规信息
            if result['violations']:
                vio_names = {'single_foot': 'Single Foot', 'double_swing': 'Double Swing', 
                            'out_of_bounds': 'Out of Bounds'}
                vio_y = y2 + 25
                for vio in result['violations']:
                    vio_text = f"! {vio_names.get(vio, vio)}"
                    cv2.putText(frame, vio_text, (x1, vio_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    vio_y += 32
        
        # 显示帧信息
        info = f"Frame: {self.frame_count}/{self.total_frames}  Time: {self.frame_count/self.fps:.1f}s"
        cv2.putText(frame, info, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, info, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        
        return frame
        
    def update_info(self, results):
        """更新右侧统计信息显示"""
        self.info_text.delete(1.0, tk.END)
        
        # 计算总人数和总跳绳次数
        total = sum(r['jump_count'] for r in results)
        
        # 显示基本信息
        self.info_text.insert(tk.END, "="*40 + "\n")
        self.info_text.insert(tk.END, f"Detected Persons: {len(results)}\n")
        self.info_text.insert(tk.END, f"Total Jumps: {total}\n")
        self.info_text.insert(tk.END, f"Current Frame: {self.frame_count}/{self.total_frames}\n")
        self.info_text.insert(tk.END, "="*40 + "\n\n")

        # 如果没有校准数据，显示提示信息
        if not self.position_mapper.is_calibrated:
            self.info_text.insert(tk.END, "No calibration loaded. Position stats unavailable.\n\n")
        
        # 显示每个人的详细信息
        for r in sorted(results, key=lambda x: x['person_id']):
            pid = r['person_id']
            status = "[ACTIVE]" if r['is_timing'] else "[PAUSED]"
            
            self.info_text.insert(tk.END, f"Person #{pid:02d} {status}\n")
            self.info_text.insert(tk.END, f"  Jumps: {r['jump_count']:3d}\n")
            self.info_text.insert(tk.END, f"  Time:  {r['elapsed_time']:5.1f}s\n")
            
            # 显示违规信息
            vio = sum(r['violation_count'].values())
            if vio > 0:
                self.info_text.insert(tk.END, f"  Violations: {vio}\n")
                if r['violation_count']['single_foot'] > 0:
                    self.info_text.insert(tk.END, f"    - Single Foot: {r['violation_count']['single_foot']}\n")
                if r['violation_count']['double_swing'] > 0:
                    self.info_text.insert(tk.END, f"    - Double Swing: {r['violation_count']['double_swing']}\n")
                if r['violation_count']['out_of_bounds'] > 0:
                    self.info_text.insert(tk.END, f"    - Out of Bounds: {r['violation_count']['out_of_bounds']}\n")
            
            self.info_text.insert(tk.END, "\n")
        
    def render_frame(self, frame, results):
        """渲染帧到画布上"""
        if frame is None:
            return

        # 转换颜色空间（BGR到RGB）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 调整图像大小以适应画布
        if canvas_w > 1 and canvas_h > 1:
            scale = min(canvas_w / w, canvas_h / h) * 0.95
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))

        # 转换为PIL图像并创建PhotoImage
        img = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=img)

        # 在画布上显示图像
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=photo)
        self.canvas.image = photo  # 保持引用防止垃圾回收
        
        # 更新统计信息
        self.update_info(results)

    def toggle_play(self):
        """切换播放/暂停状态"""
        if not self.cap or self.is_camera:
            return

        if self.is_playing:
            # 暂停播放
            self.is_playing = False
            self.play_btn.config(text="Play", bg='#3498db')
            self.status_label.config(text="Paused", fg='#e67e22')
            self.stop_pipeline()
        else:
            # 开始播放
            self.is_playing = True
            self.play_btn.config(text="Pause", bg='#e74c3c')
            self.status_label.config(text="Playing...", fg='#27ae60')
            self.start_pipeline(mode='video')

    def next_frame(self):
        """显示下一帧（单步模式）"""
        if self.is_camera or not self.cap:
            return
        if self.is_playing:
            self.toggle_play()
        if self.read_frame():
            frame_vis, results = self.analyze_frame(self.current_frame)
            self.render_frame(frame_vis, results)
            self.status_label.config(text=f"Frame {self.frame_count}/{self.total_frames}", fg='#27ae60')
        else:
            self.status_label.config(text="End of video", fg='#95a5a6')

    def reset(self):
        """重置到初始状态"""
        if self.is_camera:
            messagebox.showinfo("Info", "Stop the camera before resetting.")
            return

        self.stop_pipeline()
        self.is_playing = False
        self.play_btn.config(text="Play", bg='#3498db')
        
        # 重置视频到开始位置
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        
        # 重新初始化检测器
        self.detector = MultiPersonJumpRopeDetector(max_persons=20)
        self.reset_position_stats()
        self.result_written = False
        
        # 重新读取第一帧
        if self.cap and self.read_frame():
            frame_vis, results = self.analyze_frame(self.current_frame)
            self.render_frame(frame_vis, results)
        self.status_label.config(text="Reset complete", fg='#16a085')


def main():
    """主函数：创建并运行应用程序"""
    root = tk.Tk()
    app = JumpRopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # 程序入口点
    main()