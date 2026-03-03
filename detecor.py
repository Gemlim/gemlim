"""
跳绳检测核心模块
实现单人跳绳计数和违规行为检测
"""
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional


class PersonTracker:
    """单人跳绳追踪器"""
    
    def __init__(self, person_id: int, boundary_box: Optional[Tuple[int, int, int, int]] = None, debug: bool = False, enhanced_detection: bool = True):
        self.person_id = person_id
        self.jump_count = 0
        self.debug = debug  # 调试模式开关
        self.enhanced_detection = enhanced_detection  # 增强检测模式
        self.violation_count = {
            'single_foot': 0,
            'double_swing': 0,
            'out_of_bounds': 0
        }
        
        # 跳跃检测相关 - 使用头部高度（更稳定，不易被遮挡）
        self.head_height_history = deque(maxlen=60)  # 头部高度历史（用于跳跃检测）
        self.foot_height_history = deque(maxlen=60)  # 脚部高度历史
        self.center_mass_history = deque(maxlen=60)  # 身体重心历史（增强检测）
        self.velocity_history = deque(maxlen=30)  # 垂直速度历史（仅用于违规检测）
        self.jump_state = 'ground'  # 'ground' or 'air'
        self.last_jump_frame = -30
        self.jump_start_frame = 0  # 起跳帧
        self.jump_air_frames = 0  # 空中帧数
        self.jump_height = 0  # 跳跃高度（像素）
        
        # 绳子摇动检测（基于手腕运动）
        self.wrist_position_history = deque(maxlen=30)
        self.rope_swing_count = 0
        self.in_jump_swing_count = 0
        
        # 单脚跳检测
        self.left_foot_heights = deque(maxlen=10)
        self.right_foot_heights = deque(maxlen=10)
        
        # 边界框（用于出界检测和动态阈值）
        self.boundary_box = boundary_box  # (x1, y1, x2, y2) - 已废弃，保留用于向后兼容
        self.person_height = 0  # 人物高度（像素）
        self.bbox_history = deque(maxlen=10)  # 保存边界框历史用于计算平均高度
        
        # 位置跟踪（用于出界检测）
        self.initial_center_x = None  # 初始中心X坐标
        self.initial_center_y = None  # 初始中心Y坐标
        self.center_history = deque(maxlen=30)  # 中心点历史
        self.out_of_bounds_threshold = 100  # 出界阈值（像素），左右偏移超过此值判定为出界
        
        # 号码圈位置（用于脚部出圈检测）
        self.number_circle_center = None  # 号码圈中心 (x, y)
        self.number_circle_radius = 30  # 号码圈半径（像素），会在设置时动态调整
        
        # 帧计数器
        self.frame_count = 0
        
        # 计时相关（独立计时）
        self.start_time = None  # 当前这段跳绳的开始时间
        self.elapsed_time = 0.0  # 当前显示的总时间（秒）
        self.accumulated_time = 0.0  # 累计的历史时间（秒，用于多段累加）
        self.is_timing = False  # 是否正在计时
        self.last_jump_time = None  # 上次跳跃的时间
        self.stop_threshold = 3.0  # 停止阈值：3秒内没有跳跃则认为停止
        
        # 防止误检测机制（避免准备阶段误判）
        self.warmup_jumps = []  # 预热期的跳跃时间列表
        self.warmup_required = 0  # 禁用预热期，第一次跳跃就计数（避免丢失计数）
        self.warmup_time_window = 3.0  # 预留参数（当前未使用）
        
    def update(self, keypoints: np.ndarray, confidence: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None, current_time: float = 0.0) -> Dict:
        """
        更新追踪器状态
        keypoints: shape (17, 2) - YOLO pose 关键点坐标
        confidence: shape (17,) - 关键点置信度
        
        YOLO pose 关键点索引:
        0: 鼻子, 1-2: 眼睛, 3-4: 耳朵
        5-6: 肩膀, 7-8: 肘部, 9-10: 手腕
        11-12: 髋部, 13-14: 膝盖, 15-16: 脚踝
        """
        self.frame_count += 1
        violations = []
        
        # 更新边界框并计算人物高度（用于动态阈值）
        current_center_x = None
        current_center_y = None
        if bbox is not None:
            self.last_bbox = bbox  # 保存最后的bbox
            x1, y1, x2, y2 = bbox
            person_height = y2 - y1
            self.bbox_history.append(person_height)
            # 使用最近10帧的平均高度
            if len(self.bbox_history) > 0:
                self.person_height = np.mean(list(self.bbox_history))
            
            # 计算当前中心点（使用bbox中心）
            current_center_x = (x1 + x2) / 2
            current_center_y = (y1 + y2) / 2
            self.center_history.append((current_center_x, current_center_y))
            
            # 检测站定状态，记录站定时的位置作为初始位置
            if self.initial_center_x is None and len(self.center_history) >= 20:
                # 检查最近20帧的位置变化
                recent_centers = list(self.center_history)[-20:]
                x_positions = [c[0] for c in recent_centers]
                y_positions = [c[1] for c in recent_centers]
                
                # 计算位置变化范围
                x_range = max(x_positions) - min(x_positions)
                y_range = max(y_positions) - min(y_positions)
                
                # 判断是否站定：横向移动 < 40px，纵向移动 < 40px（放宽条件，更容易建立基准）
                is_standing = (x_range < 40 and y_range < 40)
                
                if is_standing:
                    # 站定状态：取这段时间的平均位置作为初始位置
                    self.initial_center_x = np.mean(x_positions)
                    self.initial_center_y = np.mean(y_positions)
                    if self.debug:
                        print(f"[DEBUG] ID {self.person_id}: 检测到站定状态，初始位置 X={self.initial_center_x:.1f}, Y={self.initial_center_y:.1f}")
        
        # 获取关键点
        nose = keypoints[0] if confidence[0] > 0.5 else None
        left_eye = keypoints[1] if confidence[1] > 0.5 else None
        right_eye = keypoints[2] if confidence[2] > 0.5 else None
        left_ankle = keypoints[15] if confidence[15] > 0.5 else None
        right_ankle = keypoints[16] if confidence[16] > 0.5 else None
        left_wrist = keypoints[9] if confidence[9] > 0.5 else None
        right_wrist = keypoints[10] if confidence[10] > 0.5 else None
        
        # 记录头部高度（用于跳跃检测 - 优先级：鼻子 > 双眼平均）
        head_height = None
        if nose is not None:
            head_height = nose[1]
        elif left_eye is not None and right_eye is not None:
            head_height = (left_eye[1] + right_eye[1]) / 2
        elif left_eye is not None:
            head_height = left_eye[1]
        elif right_eye is not None:
            head_height = right_eye[1]
        
        if head_height is not None:
            self.head_height_history.append(head_height)
        
        # 计算脚部平均高度（Y坐标越大表示越低）
        avg_foot_height = None
        if left_ankle is not None and right_ankle is not None:
            avg_foot_height = (left_ankle[1] + right_ankle[1]) / 2
            self.foot_height_history.append(avg_foot_height)
            
            # 保存单个脚的高度用于单脚跳检测
            self.left_foot_heights.append(left_ankle[1])
            self.right_foot_heights.append(right_ankle[1])
        
        # 增强检测：计算身体重心（头部+髋部的平均）
        if self.enhanced_detection:
            left_hip = keypoints[11] if confidence[11] > 0.5 else None
            right_hip = keypoints[12] if confidence[12] > 0.5 else None
            
            # 计算重心Y坐标
            center_mass_y = None
            valid_points = []
            if head_height is not None:
                valid_points.append(head_height)
            if left_hip is not None:
                valid_points.append(left_hip[1])
            if right_hip is not None:
                valid_points.append(right_hip[1])
            
            if len(valid_points) >= 2:  # 至少需要2个有效点
                center_mass_y = np.mean(valid_points)
                self.center_mass_history.append(center_mass_y)
                
                # 计算垂直速度（像素/帧）
                if len(self.center_mass_history) >= 2:
                    velocity = self.center_mass_history[-2] - self.center_mass_history[-1]  # 向上为正
                    self.velocity_history.append(velocity)
        
        # 检测跳跃
        jump_detected = self._detect_jump()
        if jump_detected:
            # 检查当前跳跃是否出界（在跳跃计数前检查）
            is_out_of_bounds = False
            if self.jump_count >= 3 and self.is_timing:
                if self._check_feet_out_of_circle(left_ankle, right_ankle):
                    is_out_of_bounds = True
                    # 不要重复累加出界违规（已经在前面143行检查过了）
            
            # 如果还未开始正式计时（预热期）
            if not self.is_timing:
                # 如果禁用预热期（warmup_required=0），直接开始计时
                if self.warmup_required == 0:
                    self.start_time = current_time
                    self.is_timing = True
                    self.jump_count = 1  # 第一次跳跃
                    self.last_jump_time = current_time
                else:
                    # 使用预热期机制
                    # 记录预热期的跳跃
                    self.warmup_jumps.append(current_time)
                    
                    # 清理超出时间窗口的旧跳跃记录
                    self.warmup_jumps = [t for t in self.warmup_jumps 
                                        if current_time - t <= self.warmup_time_window]
                    
                    # 检查是否满足开始条件：在时间窗口内有足够的连续跳跃
                    if len(self.warmup_jumps) >= self.warmup_required:
                        # 满足条件，正式开始计时（从第一次有效跳跃开始）
                        self.start_time = self.warmup_jumps[0]
                        self.is_timing = True
                        self.jump_count += len(self.warmup_jumps)  # 累加预热期的所有跳跃（修复：从重置改为累加）
                        self.last_jump_time = current_time
                    # else: 还不满足条件，继续预热期
            else:
                # 已经在计时，检查是否出界
                if not is_out_of_bounds:
                    # 出界的跳跃不计数
                    self.jump_count += 1
                    self.last_jump_time = current_time
                else:
                    # 出界跳跃：不计数，但记录时间（避免停止计时）
                    self.last_jump_time = current_time
                    # print(f"[DEBUG] ID {self.person_id}: 出界跳跃，不计数")
                
                # 只在正式计时后才检测违规（避免预热期误判）
                # 检测是否为单脚跳
                if self._check_single_foot_jump():
                    violations.append('single_foot')
                    self.violation_count['single_foot'] += 1
                
                # 检测一跳多摇
                if self._check_double_swing():
                    violations.append('double_swing')
                    self.violation_count['double_swing'] += 1
            
            self.last_jump_frame = self.frame_count
            self.in_jump_swing_count = 0
        
        # 检测是否停止跳绳（基于视频时间）
        if self.is_timing and self.last_jump_time is not None:
            time_since_last_jump = current_time - self.last_jump_time
            
            if time_since_last_jump > self.stop_threshold:
                # 超过阈值，停止计时
                self.is_timing = False
                # 累加本段时间到总时间
                segment_time = self.last_jump_time - self.start_time
                self.accumulated_time += segment_time
                self.elapsed_time = self.accumulated_time
                self.warmup_jumps = []  # 清空预热记录，为下次重新开始做准备
                if self.debug:
                    print(f"[DEBUG] ID {self.person_id}: 停止计时，本段时长={segment_time:.2f}秒，累计总时长={self.elapsed_time:.2f}秒")
        
        # 更新当前已用时间（基于视频时间）
        if self.is_timing and self.start_time is not None:
            self.elapsed_time = self.accumulated_time + (current_time - self.start_time)
        
        # 追踪手腕运动以检测绳子摇动
        if left_wrist is not None and right_wrist is not None:
            wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2
            self.wrist_position_history.append(wrist_avg_y)
            self._detect_rope_swing()
        
        return {
            'jump_count': self.jump_count,
            'elapsed_time': self.elapsed_time,
            'is_timing': self.is_timing,
            'violations': violations,
            'violation_count': self.violation_count.copy()
        }
    
    def _detect_jump(self) -> bool:
        """检测跳跃动作（临时回退到脚踝检测）"""
        # 临时使用脚踝检测，如果头部检测失败
        
        # 选择检测源：增强模式使用重心，否则使用头部
        if self.enhanced_detection and len(self.center_mass_history) >= 10:
            recent_heights = list(self.center_mass_history)
            detection_source = "重心"
        else:
            recent_heights = list(self.head_height_history)
            detection_source = "头部"
        
        if len(recent_heights) < 10:
            return False
        
        # 改进的基准线计算：使用更长的时间窗口和百分位数
        if self.enhanced_detection and len(recent_heights) >= 30:
            # 使用75分位数作为基准（更稳定，不易受跳跃影响）
            ground_level = np.percentile(recent_heights[-30:], 75)
        else:
            # 原始方法：取最近10帧中最大的5个的平均
            max_heights = sorted(recent_heights[-10:], reverse=True)[:5]
            ground_level = np.mean(max_heights)
        
        # 当前高度
        current_height = recent_heights[-1]
        
        # 动态计算阈值（根据人物大小）
        if self.enhanced_detection:
            # 增强模式：使用更保守的阈值
            base_jump_threshold = 6  # 降低阈值，提高灵敏度
        else:
            base_jump_threshold = 8  # 原始阈值
        
        standard_height = 300  # 标准人物高度（像素）
        
        if self.person_height > 0:
            # 动态调整阈值
            scale_factor = self.person_height / standard_height
            jump_threshold = base_jump_threshold * scale_factor
            land_threshold = jump_threshold * 2.0  # 落地阈值为起跳的2倍（更合理的比例）
        else:
            # 如果没有边界框信息，使用固定阈值
            jump_threshold = base_jump_threshold
            land_threshold = base_jump_threshold * 2.0  # 保持2倍关系
        
        # 判断使用的检测方式
        detection_method = "头部" if len(self.head_height_history) >= 10 else "脚踝"
        
        if self.jump_state == 'ground':
            # 检测起跳：高度明显上升（Y坐标减小）
            height_diff = ground_level - current_height
            
            # 起跳判定：超过阈值
            should_jump = height_diff > jump_threshold
            
            # 增强检测：添加速度验证
            if self.enhanced_detection and len(self.velocity_history) >= 3:
                # 检查是否有向上的速度（velocity > 0表示向上）
                recent_velocity = list(self.velocity_history)[-3:]
                avg_velocity = np.mean(recent_velocity)
                # 要求有明显的向上速度（至少1像素/帧）
                has_upward_velocity = avg_velocity > 1.0
                should_jump = should_jump and has_upward_velocity
                
                if self.debug and height_diff > jump_threshold:
                    print(f"[DEBUG] 速度检查: avg_vel={avg_velocity:.2f}, 向上={has_upward_velocity}")
            
            if should_jump:
                self.jump_state = 'air'
                self.jump_start_frame = self.frame_count  # 记录起跳帧
                self.jump_air_frames = 0  # 重置空中帧数
                if self.debug:
                    vel_info = f", 速度={np.mean(list(self.velocity_history)[-3:]):.2f}" if self.enhanced_detection else ""
                    print(f"[DEBUG] 起跳({detection_source}): ground={ground_level:.1f}, current={current_height:.1f}, diff={height_diff:.1f}, 阈值={jump_threshold:.1f}{vel_info}")
                    
        elif self.jump_state == 'air':
            # 累加空中帧数
            self.jump_air_frames += 1
            
            # 检测落地：头部回到基准位置（Y坐标增大）
            height_diff = ground_level - current_height
            
            # 每5帧输出一次空中状态（调试用）
            if self.debug and self.jump_air_frames % 5 == 0:
                print(f"[DEBUG] 空中({detection_method}): 第{self.jump_air_frames}帧, diff={height_diff:.1f}, 落地阈值={land_threshold:.1f}")
            
            # 落地判定：头部接近基准位置 OR 空中时间过长（超时保护）
            # 正常跳跃空中时间：3-15帧（0.1-0.5秒 @ 30fps）
            # 如果超过30帧还没落地，强制认为落地（避免卡住不计数，同时支持一跳多摇）
            is_landed = (height_diff <= land_threshold) or (self.jump_air_frames > 30)
            
            if is_landed:
                # 判断落地原因
                land_reason = "超时" if self.jump_air_frames > 30 else "正常"
                if self.debug:
                    if land_reason == "超时":
                        print(f"[DEBUG] 落地(超时): 空中{self.jump_air_frames}帧 > 30帧，强制落地")
                    else:
                        print(f"[DEBUG] 落地(正常): diff={height_diff:.1f} <= {land_threshold:.1f}, 空中{self.jump_air_frames}帧")
                
                self.jump_state = 'ground'
                
                # 防止重复计数（增加间隔避免同一跳跃被重复计数）
                if self.frame_count - self.last_jump_frame > 8:  # 8帧（约0.27秒@30fps）
                    # 计算跳跃高度（头部最高点到基准线的距离）
                    max_jump_height = ground_level - min(recent_heights[-10:])
                    self.jump_height = max_jump_height
                    
                    # 额外验证：区分跳绳和行走
                    if self._verify_is_jump(max_jump_height, jump_threshold):
                        if self.debug:
                            print(f"[DEBUG] ✅ 检测到跳绳({detection_method})！跳跃={max_jump_height:.1f}px, 空中={self.jump_air_frames}帧, 阈值={jump_threshold:.1f}px (人物={self.person_height:.0f}px)")
                        return True
                    else:
                        if self.debug:
                            print(f"[DEBUG] ⚠️ 疑似行走，未计数。跳跃={max_jump_height:.1f}px, 空中={self.jump_air_frames}帧")
        
        return False
    
    def _verify_is_jump(self, jump_height: float, threshold: float) -> bool:
        """验证是否为真正的跳跃（而非行走） - 使用头部跳跃高度"""
        
        # 验证1：跳跃高度必须达到阈值（放宽要求以匹配参考值）
        if jump_height < threshold * 1.0:  # 达到阈值即可（放宽）
            return False
        
        # 验证2：空中时间必须合理（放宽范围）
        if self.jump_air_frames < 1 or self.jump_air_frames > 25:
            return False
        
        # 验证3：检查横向移动（只排除特别明显的行走）
        if len(self.center_history) >= 10:
            recent_centers = list(self.center_history)[-10:]
            x_positions = [c[0] for c in recent_centers]
            x_range = max(x_positions) - min(x_positions)
            
            # 横向移动限制（从80%降到50%，更严格过滤行走）
            if self.person_height > 0:
                max_x_movement = self.person_height * 0.5  # 人物高度的50%
            else:
                max_x_movement = 100  # 默认阈值
            
            if x_range > max_x_movement:
                if self.debug:
                    print(f"[DEBUG]   横向移动过大: {x_range:.1f}px > {max_x_movement:.1f}px (疑似行走)")
                return False
        
        return True
    
    def _check_recent_rope_swing(self) -> bool:
        """检查最近是否有摇绳动作（动态阈值）"""
        if len(self.wrist_position_history) < 10:
            return False
        
        recent_wrists = list(self.wrist_position_history)[-10:]
        
        # 检查手腕是否有上下运动
        wrist_range = max(recent_wrists) - min(recent_wrists)
        
        # 动态调整手腕摇动阈值
        base_wrist_threshold = 20  # 基础阈值
        standard_height = 300
        
        if self.person_height > 0:
            scale_factor = self.person_height / standard_height
            wrist_threshold = base_wrist_threshold * scale_factor
        else:
            wrist_threshold = base_wrist_threshold
        
        # 检查手腕移动是否超过动态阈值
        if wrist_range > wrist_threshold:
            if self.debug:
                print(f"[DEBUG]   手腕移动范围: {wrist_range:.1f}px > {wrist_threshold:.1f}px (阈值)")
            return True
        else:
            if self.debug:
                print(f"[DEBUG]   手腕移动不足: {wrist_range:.1f}px < {wrist_threshold:.1f}px")
            return False
    
    def _detect_rope_swing(self):
        """检测绳子摇动（基于手腕上下运动）"""
        if len(self.wrist_position_history) < 6:
            return
        
        recent_wrists = list(self.wrist_position_history)[-6:]
        
        # 检测手腕的周期性运动
        # 简化方法：检测手腕Y坐标的波峰和波谷
        if len(recent_wrists) >= 6:
            # 检测上升然后下降的模式
            mid_point = recent_wrists[3]
            prev_points = recent_wrists[:3]
            if all(mid_point < p for p in prev_points):
                # 检测到一个波峰
                if self.jump_state == 'air':
                    self.in_jump_swing_count += 1
    
    def _check_single_foot_jump(self) -> bool:
        """检测单脚跳（两只脚高度差异过大）"""
        # 需要至少5帧的数据来判断
        if len(self.left_foot_heights) < 5 or len(self.right_foot_heights) < 5:
            return False
        
        # 获取最近5帧的脚部高度
        left_heights = list(self.left_foot_heights)[-5:]
        right_heights = list(self.right_foot_heights)[-5:]
        
        # 计算两只脚的平均高度差异
        height_diff = abs(np.mean(left_heights) - np.mean(right_heights))
        
        # 动态阈值：根据人物高度调整
        # 一般人的单脚跳，两脚高度差会超过人物高度的10%
        threshold = max(30, self.person_height * 0.1) if self.person_height > 0 else 30
        
        # 如果两只脚高度差异过大，判定为单脚跳
        if height_diff > threshold:
            if self.debug:
                print(f"[违规] 人物 {self.person_id}: 检测到单脚跳 (脚部高度差={height_diff:.1f}px > 阈值={threshold:.1f}px)")
            return True
        
        return False
    
    def _check_double_swing(self) -> bool:
        """检测一跳多摇（基于空中滞留时间）"""
        # 一跳多摇的特征：
        # 1. 跳得更高
        # 2. 空中滞留时间更长
        # 3. 绳子在一次跳跃中过2次或更多次
        
        # 假设视频为30fps，正常跳跃空中时间约3-8帧（0.1-0.27秒）
        # 一跳多摇需要更长的空中时间，通常 >= 10帧（>= 0.33秒 @ 30fps）
        
        # 动态阈值：根据人物高度调整
        # 人物越大，正常跳跃高度也越高，需要更高的阈值
        air_time_threshold = 10  # 基础阈值：10帧（进一步降低以提高检测灵敏度）
        
        # 同时需要跳跃高度超过正常水平
        # 正常跳跃高度约为人物高度的4-7%
        # 一跳多摇跳跃高度通常 >= 人物高度的8%
        if self.person_height > 0:
            height_threshold = self.person_height * 0.08  # 8%的人物高度（进一步降低以提高检测灵敏度）
        else:
            height_threshold = 28  # 默认阈值（进一步降低）
        
        # 判断：空中滞留时间长 且 跳跃高度高
        if self.jump_air_frames >= air_time_threshold and self.jump_height >= height_threshold:
            if self.debug:
                print(f"[违规] 人物 {self.person_id}: 检测到一跳多摇 (空中={self.jump_air_frames}帧, 跳高={self.jump_height:.1f}px > 阈值={height_threshold:.1f}px)")
            return True
        
        return False
    
    
    def _check_feet_out_of_circle(self, left_ankle, right_ankle) -> bool:
        """
        检测脚部是否出界（基于初始位置）
        
        检测逻辑：
        - 使用人物bbox的初始中心位置作为基准
        - 检测脚踝是否偏离初始位置太远
        - 只在开始跳绳后检测（前3次跳跃建立基准）
        """
        # 如果还没有建立初始位置基准，不检测
        if self.initial_center_x is None or self.jump_count < 3:
            return False
        
        # 如果没有检测到脚踝，不检测
        if left_ankle is None and right_ankle is None:
            return False
        
        # 出界阈值：100像素（左右偏移）
        threshold = self.out_of_bounds_threshold
        
        # 检查左脚是否出界
        if left_ankle is not None:
            ankle_x = left_ankle[0]
            x_offset = abs(ankle_x - self.initial_center_x)
            
            if x_offset > threshold:
                if self.debug:
                    print(f"[违规] 人物 {self.person_id}: 左脚出界 (偏移={x_offset:.1f}px > 阈值={threshold}px)")
                return True
        
        # 检查右脚是否出界
        if right_ankle is not None:
            ankle_x = right_ankle[0]
            x_offset = abs(ankle_x - self.initial_center_x)
            
            if x_offset > threshold:
                if self.debug:
                    print(f"[违规] 人物 {self.person_id}: 右脚出界 (偏移={x_offset:.1f}px > 阈值={threshold}px)")
                return True
        
        return False
    
    def set_number_circle(self, center: Tuple[float, float], radius: float):
        """设置号码圈位置和半径"""
        self.number_circle_center = center
        self.number_circle_radius = radius
    
    def set_out_of_bounds_threshold(self, threshold: float):
        """设置出界阈值（像素）"""
        self.out_of_bounds_threshold = threshold
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'person_id': self.person_id,
            'jump_count': self.jump_count,
            'elapsed_time': self.elapsed_time,
            'is_timing': self.is_timing,
            'violations': self.violation_count.copy(),
            'total_violations': sum(self.violation_count.values())
        }


class MultiPersonJumpRopeDetector:
    """多人跳绳检测器"""
    
    def __init__(self, max_persons: int = 10, debug: bool = False, enhanced_detection: bool = True):
        self.max_persons = max_persons
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_person_id = 1  # ID从1开始
        self.debug = debug  # 调试模式开关
        self.enhanced_detection = enhanced_detection  # 增强检测模式
        
        # 用于人员匹配的历史位置
        self.person_positions: Dict[int, deque] = {}
        
        # 出界检测配置
        self.boundary_box: Optional[Tuple[int, int, int, int]] = None  # 已废弃
        self.out_of_bounds_threshold = 100  # 出界检测阈值（像素）
    
    def set_boundary_box(self, boundary_box: Optional[Tuple[int, int, int, int]]):
        """设置出界检测边界框（已废弃，保留用于向后兼容）"""
        self.boundary_box = boundary_box
        # 更新所有已有追踪器的边界框
        for tracker in self.trackers.values():
            tracker.boundary_box = boundary_box
    
    def set_out_of_bounds_threshold(self, threshold: float):
        """设置出界检测阈值（像素）"""
        # 更新所有已有追踪器的阈值
        for tracker in self.trackers.values():
            tracker.set_out_of_bounds_threshold(threshold)
        
    def update(self, detections: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]], current_time: float = 0.0) -> List[Dict]:
        """
        更新所有检测到的人
        detections: List of (keypoints, confidence, bbox)
            - keypoints: (17, 2) 关键点坐标
            - confidence: (17,) 关键点置信度
            - bbox: (x1, y1, x2, y2) 边界框
        current_time: 当前视频时间（秒），用于计时
        
        Returns: List of detection results for each person
        """
        results = []
        
        # 匹配检测到的人与已有追踪器
        matched_ids = self._match_persons(detections)
        
        # 更新每个追踪器
        for detection, person_id in zip(detections, matched_ids):
            # 跳过被忽略的检测（person_id == -1）
            if person_id == -1:
                continue
                
            keypoints, confidence, bbox = detection
            
            # 如果是新人，创建追踪器
            if person_id not in self.trackers:
                if len(self.trackers) < self.max_persons:
                    new_tracker = PersonTracker(person_id, boundary_box=self.boundary_box, debug=self.debug, enhanced_detection=self.enhanced_detection)
                    new_tracker.set_out_of_bounds_threshold(self.out_of_bounds_threshold)  # 设置阈值
                    self.trackers[person_id] = new_tracker
                    self.person_positions[person_id] = deque(maxlen=30)
                else:
                    continue  # 超过最大人数限制
            
            # 更新追踪器（传递bbox用于动态阈值计算，current_time用于视频时间计时）
            result = self.trackers[person_id].update(keypoints, confidence, bbox, current_time)
            result['person_id'] = person_id
            result['bbox'] = bbox
            
            # 更新位置历史
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.person_positions[person_id].append(center)
            
            results.append(result)
        
        return results
    
    def _match_persons(self, detections: List) -> List[int]:
        """匹配检测到的人与已有追踪器"""
        if len(self.trackers) == 0:
            # 首次检测：按行分组，每行内从左到右排序
            # 下方（Y大）: ID 1-5
            # 上方（Y小）: ID 6-10
            
            # 创建(center_x, center_y, detection对象)的列表
            detection_with_pos = []
            for detection in detections:
                keypoints, confidence, bbox = detection
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                detection_with_pos.append((center_x, center_y, detection))
            
            # 过滤围观者，保留最多15人（确保11个点位都能被检测到）
            selected_detections = detection_with_pos
            if len(detection_with_pos) > 15:
                # 按Y坐标排序，取中间的15个（过滤边缘的围观者）
                sorted_by_y = sorted(detection_with_pos, key=lambda x: x[1])
                start_idx = (len(sorted_by_y) - 15) // 2
                selected_detections = sorted_by_y[start_idx:start_idx + 15]
                print(f"[INFO] 检测到 {len(sorted_by_y)} 人，只保留中间区域的15人")
            
            # 按Y坐标排序，找到中位数分成上下两组
            sorted_by_y = sorted(selected_detections, key=lambda x: x[1])
            mid_idx = len(sorted_by_y) // 2
            
            # 下方（Y坐标大）- ID 1-5
            bottom_group = sorted_by_y[mid_idx:]
            # 上方（Y坐标小）- ID 6-10
            top_group = sorted_by_y[:mid_idx]
            
            # 每组内按X坐标（从左到右）排序
            bottom_group.sort(key=lambda x: x[0])
            top_group.sort(key=lambda x: x[0])
            
            # 创建映射：detection对象 -> 新ID
            detection_to_id = {}
            
            # 下方组: ID 1-5
            for i, (cx, cy, detection) in enumerate(bottom_group):
                detection_to_id[id(detection)] = i + 1
            
            # 上方组: ID 6-10
            for i, (cx, cy, detection) in enumerate(top_group):
                detection_to_id[id(detection)] = i + 6
            
            # 按原始detections顺序返回对应的ID，未被选中的返回-1
            result_ids = [detection_to_id.get(id(det), -1) for det in detections]
            self.next_person_id = len(detections) + 1
            
            return result_ids
        
        matched_ids = []
        used_ids = set()
        
        for detection in detections:
            keypoints, confidence, bbox = detection
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # 找到最近的已有追踪器
            min_distance = float('inf')
            best_id = None
            
            for person_id, position_history in self.person_positions.items():
                if person_id in used_ids:
                    continue
                
                if len(position_history) > 0:
                    last_pos = position_history[-1]
                    distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    # 动态距离阈值：根据bbox大小调整（默认150px）
                    match_threshold = 150
                    if distance < min_distance and distance < match_threshold:
                        min_distance = distance
                        best_id = person_id
            
            # 如果找到匹配的ID，使用它；否则忽略（不再创建新ID）
            if best_id is not None:
                matched_ids.append(best_id)
                used_ids.add(best_id)
            else:
                # 已达到最大人数限制，不再分配新ID，跳过该检测
                matched_ids.append(-1)  # 使用-1表示忽略
        
        return matched_ids
    
    def get_all_statistics(self) -> List[Dict]:
        """获取所有人的统计信息"""
        return [tracker.get_statistics() for tracker in self.trackers.values()]
    
    def reset(self):
        """重置所有追踪器"""
        self.trackers.clear()
        self.person_positions.clear()
        self.next_person_id = 1  # ID从1开始
