from pynput import mouse

_coords = []

def on_click(x: int, y: int, button, pressed: bool) -> None | bool:
    if pressed:
        _coords.append((x, y))
        print(f"Clicked at: {x}, {y}", flush=True)
        if len(_coords) == 2:
            return False  # 退出监听
    

def get_game_area() -> tuple:
    global _coords
    _coords = [] # 每次清空坐标列表

    print("请依次点击 游戏区域左上角 和 右下角", flush=True)
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    if len(_coords) != 2:
        raise RuntimeError("未能正确获取游戏区域坐标")
    
    (x1, y1), (x2, y2) = _coords
    left, top = min(x1, x2), min(y1, y2)
    weight, hight = abs(x2 - x1), abs(y2 - y1)

    region = (left, top, weight, hight)
    print(f"区域参数：({region})", flush=True)
    return region