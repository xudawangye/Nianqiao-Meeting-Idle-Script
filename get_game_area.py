from pynput import mouse

coords = []

def on_click(x, y, button, pressed):
    if pressed:
        coords.append((x, y))
        print(f"Clicked at: {x}, {y}", flush=True)
        if len(coords) == 2:
            # 取左上和右下
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            left, top = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            print("区域参数:", (left, top, w, h), flush=True)
            return False  # 退出监听

with mouse.Listener(on_click=on_click) as listener:
    print("请依次点击 游戏区域左上角 和 右下角", flush=True)
    listener.join()
