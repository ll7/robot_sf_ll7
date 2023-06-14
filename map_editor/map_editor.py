import os
from time import sleep
from typing import Tuple, Callable, Union, List
from threading import Thread

import turtle
import tkinter as tk
import tkinter.scrolledtext as tks

from map_editor.map_file_parser import \
    parse_mapfile_text, VisualizableMapConfig

Vec2D = Tuple[float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]


class MapCanvas:
    def __init__(self, frame: tk.Frame):
        self.canvas = tk.Canvas(frame)
        self.canvas.config(width=800, height=800)
        screen = turtle.TurtleScreen(self.canvas)
        screen.bgcolor("white")
        self.turtle = turtle.RawTurtle(screen, visible=False)
        screen.tracer(0)

    def pack(self):
        self.canvas.pack()

    def render(self, map_config: VisualizableMapConfig):
        (min_x, max_x), (min_y, max_y) = map_config.x_margin, map_config.y_margin
        width, height = max_x- min_x, max_y - min_y
        if width > height:
            max_y = min_y + width
        else:
            max_x = min_x + height
        self.turtle.screen.setworldcoordinates(min_x, min_y, max_x, max_y)
        self.turtle.clear()
        self.turtle.color('black')

        def draw_line(p1: Vec2D, p2: Vec2D):
            self.turtle.up()
            self.turtle.setpos(p1)
            self.turtle.down()
            self.turtle.setpos(p2)

        for s_x, e_x, s_y, e_y in map_config.obstacles:
            draw_line((s_x, s_y), (e_x, e_y))

        def rect_points(rect: Rect) -> List[Vec2D]:
            def add_vec(v1: Vec2D, v2: Vec2D) -> Vec2D:
                return v1[0] + v2[0], v1[1] + v2[1]
            def sub_vec(v1: Vec2D, v2: Vec2D) -> Vec2D:
                return v1[0] - v2[0], v1[1] - v2[1]
            p1, p2, p3 = rect
            p4 = add_vec(sub_vec(p3, p2), p1)
            return [p1, p2, p3, p4]

        def draw_rect(points: List[Vec2D]):
            p1, p2, p3, p4 = points
            self.turtle.up()
            self.turtle.setpos(p1)
            self.turtle.down()
            self.turtle.setpos(p2)
            self.turtle.setpos(p3)
            self.turtle.setpos(p4)
            self.turtle.setpos(p1)

        self.turtle.color('green')
        for rect in map_config.goal_zones:
            draw_rect(rect_points(rect))

        self.turtle.color('blue')
        for rect in map_config.robot_spawn_zones:
            draw_rect(rect_points(rect))

        self.turtle.color('red')
        for rect in map_config.ped_spawn_zones:
            draw_rect(rect_points(rect))

        def draw_circle(center: Vec2D, radius: float):
            self.turtle.up()
            self.turtle.setpos((center[0], center[1] - radius))
            self.turtle.down()
            self.turtle.circle(radius, steps=50)

        self.turtle.color('green')
        self.turtle.fillcolor('green')
        for route in map_config.robot_routes:
            for p in route.waypoints:
                draw_circle(p, radius=1)

        self.turtle.up()


class TextEditor:
    def __init__(self, frame: tk.Frame):
        self.input = tks.ScrolledText(frame)
        self.input.bind("<Control-Key-a>", self.select_all)

    @property
    def text(self) -> str:
        return self.input.get("1.0", 'end-1c')

    def pack(self):
        self.input.pack(side=tk.LEFT, fill="both")

    def clear_text(self):
        self.input.delete('1.0', tk.END)

    def append_text(self, text: str):
        self.input.insert(tk.END, text)

    def select_all(self, e: tk.Event):
        self.input.tag_add(tk.SEL, "1.0", tk.END)
        self.input.mark_set(tk.INSERT, "1.0")
        self.input.see(tk.INSERT)
        return 'break'


class MapEditor:
    def __init__(self):
        TITLE = "RobotSF Map Editor"
        self.master = tk.Tk()
        self.master.resizable(False, False)
        self.master.title(TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame_editor = tk.Frame(self.master)
        self.frame_canvas = tk.Frame(self.master)

        self.map_canvas = MapCanvas(self.frame_canvas)
        self.text_editor = TextEditor(self.frame_editor)

        self.is_shutdown_requested = False
        self.map_rendering_thread: Union[Thread, None] = None
        self.last_text = ''
        self._load_example_map()

    def launch(self):
        self.pack()

        def reload_map():
            config_content = self.text_editor.text
            if config_content != self.last_text:
                map_config = parse_mapfile_text(config_content)
                if map_config:
                    self.last_text = config_content
                    try:
                        self.map_canvas.render(map_config)
                    except:
                        print('data error, cannot display')
                else:
                    print('parsing config file failed!')

        def reload_map_as_daemon(frequency_hz: float, is_term: Callable[[], bool]):
            reload_intercal_secs = 1 / frequency_hz
            while not is_term():
                reload_map()
                sleep(reload_intercal_secs)

        RELOAD_FREQUENCY = 4
        args = (RELOAD_FREQUENCY, lambda: self.is_shutdown_requested)
        self.map_rendering_thread = Thread(target=reload_map_as_daemon, args=args)
        self.map_rendering_thread.start()
        self.master.mainloop()

    def pack(self):
        self.frame_canvas.pack(side=tk.RIGHT)
        self.frame_editor.pack(side=tk.LEFT, fill='both')
        self.map_canvas.pack()
        self.text_editor.pack()

    def on_closing(self):
        if self.map_rendering_thread:
            self.is_shutdown_requested = True
            self.map_rendering_thread.join()
        self.master.destroy()

    def _load_example_map(self):
        current_dir = os.path.dirname(__file__)
        example_filepath = os.path.join(current_dir, 'map_example.json')
        with open(example_filepath, 'r') as file:
            text = file.read()
        self.text_editor.clear_text()
        self.text_editor.append_text(text)
