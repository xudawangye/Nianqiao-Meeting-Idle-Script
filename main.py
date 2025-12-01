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
    #初始化模型的路径、桥的速度、失败次数
    def __init__(self):
        self.model = YOLO(r".\runs\detect\train\weights\best.pt") # 模型路径，请替换为你实际的路径
        self.speed = 120  # 桥伸长速度(像素/秒)
        self.fail_count = 0 #失败次数

    #获取距离
    def get_dis(self, show=True):
        # 截屏
        img = pyautogui.screenshot(region=(1743, 535, 656, 307)) # 根据实际窗口的位置调整，可以使用项目中的get_game_area.py脚本获取游戏区域坐标
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.ascontiguousarray(img)

        # 预测
        res = self.model.predict(img, imgsz=640, conf=0.01, verbose=False)

        human_center = None
        pillar_list = [] #(center_x, center_y, class)

        # 解析结果并可视化
        for result in res:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if int(cls) == 0:  # 陶陶
                    human_center = (cx, cy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, human_center, 5, (0, 0, 255), -1)
                elif int(cls) == 1 or int(cls) == 2:  # 1：柱子，2：x2经验包
                    pillar_list.append((cx, cy,int(cls)))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        min_dist = None
        nearest_pillar = None
        pillar_class = None

        # 算出陶陶与前方最近柱子的距离
        if human_center is not None and pillar_list:
            forward_pillars = [p for p in pillar_list if p[0] > human_center[0]]
            if forward_pillars:
                distances = [p[0] - human_center[0] for p in forward_pillars]
                min_index = np.argmin(distances)
                nearest_pillar = forward_pillars[min_index]
                min_dist = distances[min_index]
                pillar_class = forward_pillars[min_index][2]

                # 画连线
                cv2.line(img, human_center, nearest_pillar[:2], (255, 0, 0), 2)

        if show:
            cv2.imshow("Detection", img)
            if cv2.waitKey(1) == 27:
                exit(0)

        return min_dist, nearest_pillar, human_center, pillar_class


    def auto_build(self):
        pillar_count = 0
        slope_config = {
        1: [-0.330, 0.639],   # 柱子在前和右的两个方向的斜率
        2: [-0.390, 0.520],   # x2经验包在前和右的两个方向的斜率
    }

        # 建桥主循环
        while True:
            distance, pillar, human_center, pillar_class = self.get_dis(show=True)
            if distance is None or pillar is None or human_center is None:
                logging.info("没有检测到人或柱子，重试中...")
                sys.stdout.flush()
                time.sleep(0.1)
                continue

            # 计算斜率，在程序中使用斜率来判断柱子是否对齐
            x1, y1 = human_center
            x2, y2, _ = pillar 
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')

            logging.info(f"检测柱子斜率: {slope:.3f}, 距离: {distance}")
            sys.stdout.flush()


            # 判断是否对齐
            if pillar_class not in slope_config:
                continue  # 未知类别，跳过
            target_slopes = slope_config[pillar_class]

            # 判断是否匹配符合要求的斜率
            aligned = any(abs(slope - ts) < 0.035 for ts in target_slopes)
            """
                由于判断斜率的阈值写的比较严格，可能会出现柱子没有移动但斜率不符合要求的情况，
                所以在下面给了用户手动介入的功能，避免程序卡死，如果两边的柱子未动但斜率不合格，
                只需要根据提示输入y，按当前参数放板即可
            """

            logging.info(
                f"class={pillar_class}, slope={slope:.3f}, "
                f"target={target_slopes}, aligned={aligned}"
            )
            
            # 执行按键操作
            if aligned:
                self.fail_count = 0
                press_time = distance / self.speed
                if press_time > 1.85:
                    continue
                # 默认按下空格键，请在模拟器中将空格键绑在放桥的按钮上
                pydirectinput.keyDown("space")
                time.sleep(press_time)
                pydirectinput.keyUp("space")
                direction = "forward" if aligned else "right"
                logging.info(f"对齐成功（方向 {direction}）！按空格 {press_time:.3f}s")
                sys.stdout.flush()
                pillar_count += 1
                time.sleep(3.85)
            # 针对移动的柱子的处理方法
            else:
                logging.info("未对齐，继续检测...")
                sys.stdout.flush()
                time.sleep(0.05)
                self.fail_count += 1
                logging.info(f"未对齐（连续失败 {self.fail_count}/30），继续检测...")
                sys.stdout.flush()

                #引入人工判断，避免程序死循环
                if self.fail_count >= 30:
                    print("\n==============================")
                    print("⚠ 已连续 30 次未对齐")
                    print(f"当前检测斜率：{slope:.3f}")
                    print(f"目标范围：{target_slopes}")
                    print("是否强制使用当前斜率？")
                    print("输入 y = 使用当前斜率释放木板")
                    print("输入 n = 放弃本次结果，继续检测")
                    print("输入 q = 退出脚本")
                    print("==============================")

                    choice = input("请输入 y/n/q: ，点击回车后请在0.8s内将视窗焦点转到游戏画面，避免无法放板\n").strip().lower()
                    
                    if choice == "y":
                        logging.info("你选择了：使用当前斜率释放木板")
                        time.sleep(0.8)
                        self.fail_count = 0
                        press_time = distance / self.speed
                        if press_time <= 1.85:
                            pydirectinput.keyDown("space")
                            time.sleep(press_time)
                            pydirectinput.keyUp("space")
                            logging.info(f"强制对齐成功！按空格 {press_time:.3f}s")
                            pillar_count += 1
                            time.sleep(3.85)

                    elif choice == "q":
                        logging.info("退出程序。")
                        return
                    
                    elif choice == "n":
                        time.sleep(0.8)
                        self.fail_count = 0
                        logging.info("你选择了：放弃本次结果，继续检测")
                        time.sleep(0.05)
                        continue

                    else:
                        logging.info("无效输入，请输入 y/n/q")

if __name__ == "__main__":
    bridge_obj = Bridge()
    bridge_obj.auto_build()
