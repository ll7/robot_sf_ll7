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


class MapEditor:
    def __init__(self):
        TITLE = "Robot_SF Map Editor"
        self.master = tk.Tk()
        self.master.resizable(False, False)
        self.master.title(TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.map_canvas, self.my_turtle = self._init_turtle_canvas(self.master)
        self.editor_text_input = self._init_editor_input(self.master)
        self.is_shutdown_requested = False
        self.map_rendering_thread: Union[Thread, None] = None
        self.last_text = ''
        self._load_example_map()

    def launch(self):
        def reload_map():
            config_content = self.editor_text_input.get("1.0", 'end-1c')
            if config_content != self.last_text:
                map_config = parse_mapfile_text(config_content)
                if map_config:
                    self.last_text = config_content
                    try:
                        self._render_map_canvas(map_config)
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

    def on_closing(self):
        if self.map_rendering_thread:
            self.is_shutdown_requested = True
            self.map_rendering_thread.join()
        self.master.destroy()

    def _init_turtle_canvas(self, master: tk.Tk) -> Tuple[tk.Canvas, turtle.RawTurtle]:
        canvas = tk.Canvas(master)
        canvas.config(width=800, height=800)
        canvas.pack(side=tk.RIGHT)
        screen = turtle.TurtleScreen(canvas)
        screen.bgcolor("white")
        my_turtle = turtle.RawTurtle(screen, visible=False)
        screen.tracer(0)
        return canvas, my_turtle

    def _init_editor_input(self, master: tk.Tk) -> tk.Text:
        editor_text_input = tks.ScrolledText(master)
        editor_text_input.pack(side=tk.LEFT, fill='both')
        return editor_text_input

    def _render_map_canvas(self, map_config: VisualizableMapConfig):
        (min_x, max_x), (min_y, max_y) = map_config.x_margin, map_config.y_margin
        self.my_turtle.screen.setworldcoordinates(min_x, min_y, max_x, max_y)
        self.my_turtle.clear()
        self.my_turtle.color('black')

        def draw_line(p1: Vec2D, p2: Vec2D):
            self.my_turtle.up()
            self.my_turtle.setpos(p1)
            self.my_turtle.down()
            self.my_turtle.setpos(p2)

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
            self.my_turtle.up()
            self.my_turtle.setpos(p1)
            self.my_turtle.down()
            self.my_turtle.setpos(p2)
            self.my_turtle.setpos(p3)
            self.my_turtle.setpos(p4)
            self.my_turtle.setpos(p1)

        self.my_turtle.color('green')
        for rect in map_config.goal_zones:
            draw_rect(rect_points(rect))

        self.my_turtle.color('blue')
        for rect in map_config.robot_spawn_zones:
            draw_rect(rect_points(rect))

        self.my_turtle.color('red')
        for rect in map_config.ped_spawn_zones:
            draw_rect(rect_points(rect))

        def draw_circle(center: Vec2D, radius: float):
            self.my_turtle.up()
            self.my_turtle.setpos((center[0], center[1] - radius))
            self.my_turtle.down()
            self.my_turtle.circle(radius, steps=50)

        self.my_turtle.color('green')
        self.my_turtle.fillcolor('green')
        for route in map_config.robot_routes:
            for p in route.waypoints:
                draw_circle(p, radius=1)

        self.my_turtle.up()

    def _load_example_map(self):
        current_dir = os.path.dirname(__file__)
        example_filepath = os.path.join(current_dir, 'map_example.json')
        with open(example_filepath, 'r') as file:
            text = file.read()
        self.editor_text_input.insert(tk.END, text)
