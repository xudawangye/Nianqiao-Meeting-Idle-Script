from ultralytics import YOLO
import pyautogui
import numpy as np
import time
import pydirectinput
import cv2
import logging
import sys

logging.basicConfig(level=logging.INFO, force=True)
class Bridge:
    def __init__(self):
        self.model = YOLO(r".\runs\detect\train\weights\best.pt")
        self.speed = 133  # 桥伸长速度(像素/秒)

    def get_dis(self, show=True):
        # 截屏
        img = pyautogui.screenshot(region=(1881, 628, 613, 336))
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.ascontiguousarray(img)

        # 预测
        res = self.model.predict(img, imgsz=640, conf=0.01, verbose=False)

        human_center = None
        pillar_centers = []

        for result in res:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if int(cls) == 0:  # 人
                    human_center = (cx, cy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, human_center, 5, (0, 0, 255), -1)
                elif int(cls) == 1 or int(cls) == 2:  # 柱子
                    pillar_centers.append((cx, cy))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        min_dist = None
        nearest_pillar = None

        if human_center is not None and pillar_centers:
            forward_pillars = [p for p in pillar_centers if p[0] > human_center[0]]
            if forward_pillars:
                distances = [p[0] - human_center[0] for p in forward_pillars]
                min_index = np.argmin(distances)
                nearest_pillar = forward_pillars[min_index]
                min_dist = distances[min_index]

                # 画连线
                cv2.line(img, human_center, nearest_pillar, (255, 0, 0), 2)

        if show:
            cv2.imshow("Detection", img)
            if cv2.waitKey(1) == 27:
                exit(0)

        return min_dist, nearest_pillar, human_center


    def auto_build(self):
        pillar_count = 0
        baseline_forward_slope = -0.330  # 前进方向基准斜率
        baseline_right_slope = 0.639    # 向右方向特殊斜率

        while True:
            distance, pillar, human_center = self.get_dis(show=True)
            if distance is None or pillar is None or human_center is None:
                logging.info("没有检测到人或柱子，重试中...")
                sys.stdout.flush()
                time.sleep(0.1)
                continue

            # 计算斜率
            x1, y1 = human_center
            x2, y2 = pillar
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')

            logging.info(f"检测柱子斜率: {slope:.3f}, 距离: {distance}")
            sys.stdout.flush()


            # 判断是否对齐
            aligned_forward = abs(slope - baseline_forward_slope) < 0.05 if baseline_forward_slope is not None else False
            aligned_right   = abs(slope - baseline_right_slope) < 0.05 if baseline_right_slope is not None else False

            if aligned_forward or aligned_right:
                press_time = distance / self.speed
                if press_time > 1.85:
                    continue
                pydirectinput.keyDown("space")
                time.sleep(press_time)
                pydirectinput.keyUp("space")
                direction = "forward" if aligned_forward else "right"
                logging.info(f"对齐成功（方向 {direction}）！按空格 {press_time:.3f}s")
                sys.stdout.flush()
                pillar_count += 1
                time.sleep(3.6)
            else:
                logging.info("未对齐，继续检测...")
                sys.stdout.flush()
                time.sleep(0.05)

if __name__ == "__main__":
    bridge_obj = Bridge()
    bridge_obj.auto_build()
