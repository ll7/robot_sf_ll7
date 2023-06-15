import os
from time import sleep
from typing import Tuple, Callable, Union, List
from threading import Thread

import turtle
import tkinter as tk
import tkinter.scrolledtext as tks

from map_editor.map_file_parser import \
    parse_mapfile_text, VisualizableMapConfig, format_waypoints_as_json

Vec2D = Tuple[float, float]
Range2D = Tuple[float, float] # (low, high)
MapBounds = Tuple[Range2D, Range2D] # (dx, dy)
Rect = Tuple[Vec2D, Vec2D, Vec2D]


def draw_line(turtle: turtle.RawTurtle, p1: Vec2D, p2: Vec2D):
    turtle.up()
    turtle.setpos(p1)
    turtle.down()
    turtle.setpos(p2)


def rect_points(rect: Rect) -> List[Vec2D]:
    def add_vec(v1: Vec2D, v2: Vec2D) -> Vec2D:
        return v1[0] + v2[0], v1[1] + v2[1]
    def sub_vec(v1: Vec2D, v2: Vec2D) -> Vec2D:
        return v1[0] - v2[0], v1[1] - v2[1]
    p1, p2, p3 = rect
    p4 = add_vec(sub_vec(p3, p2), p1)
    return [p1, p2, p3, p4]


def draw_rect(turtle: turtle.RawTurtle, points: List[Vec2D]):
    p1, p2, p3, p4 = points
    turtle.up()
    turtle.setpos(p1)
    turtle.down()
    turtle.setpos(p2)
    turtle.setpos(p3)
    turtle.setpos(p4)
    turtle.setpos(p1)


def draw_circle(turtle: turtle.RawTurtle, center: Vec2D, radius: float):
    turtle.up()
    turtle.setpos((center[0], center[1] - radius))
    turtle.down()
    turtle.circle(radius, steps=50)


class RouteBuilder:
    def __init__(
            self, frame: tk.Frame, canvas: tk.Canvas,
            render_waypoints: Callable[[List[Vec2D]], None],
            fetch_map_bounds: Callable[[], MapBounds]):
        self.canvas = canvas
        self.render_waypoints = render_waypoints
        self.fetch_map_bounds = fetch_map_bounds

        self.btn_new_route = tk.Button(frame, text="New Route", command=self.new_route)
        self.btn_undo = tk.Button(frame, text="Undo", command=self.undo_waypoint)
        self.route_as_text = tk.StringVar(frame, "[]")
        self.text_route = tk.Entry(frame, textvariable=self.route_as_text)
        self.text_route.config(state=tk.DISABLED)

        self.mouse_pos = (0, 0)
        self.last_config: VisualizableMapConfig
        self.waypoints: List[Vec2D] = []

        def track_mouse_pos(event):
            self.mouse_pos = (event.x, event.y)

        self.canvas.bind('<Motion>', track_mouse_pos)
        self.canvas.bind('<Button-1>', lambda e: self.add_waypoint(self.mouse_pos))

    def pack(self):
        self.btn_undo.pack(side=tk.RIGHT)
        self.btn_new_route.pack(side=tk.RIGHT)
        self.text_route.pack(fill="x")

    def new_route(self):
        self.waypoints = []
        self._update_ui()

    def add_waypoint(self, p: Vec2D):
        bounds = self.fetch_map_bounds()
        if not bounds:
            print("WARNING: no map bounds specified!")
            return

        x_margin, y_margin = bounds
        # TODO: convert canvas mouse position to map coord

    def undo_waypoint(self):
        if len(self.waypoints) > 0:
            self.waypoints.pop()
            self._update_ui()

    def _update_ui(self):
        new_text = format_waypoints_as_json(self.waypoints)
        self.route_as_text.set(new_text)
        self.render_waypoints(self.waypoints)


class MapCanvas:
    BG_COLOR = "white"
    OBSTACLE_COLOR = "black"
    ROBOT_GOAL_COLOR = "green"
    ROBOT_SPAWN_COLOR = "blue"
    PED_SPAWN_COLOR = "red"
    ROBOT_ROUTE_COLOR = "green"
    TEMP_ROUTE_COLOR = "orange"

    def __init__(self, frame: tk.Frame):
        self.canvas = tk.Canvas(frame)
        self.canvas.config(width=800, height=800)
        screen = turtle.TurtleScreen(self.canvas)
        screen.bgcolor(MapCanvas.BG_COLOR)
        self.turtle = turtle.RawTurtle(screen, visible=False)
        screen.tracer(0)
        self.temp_waypoints = []

    def pack(self):
        self.canvas.pack()

    def render_temp_waypoints(self, waypoints: List[Vec2D]):
        self.temp_waypoints = waypoints

    def render(self, map_config: VisualizableMapConfig):
        (min_x, max_x), (min_y, max_y) = map_config.x_margin, map_config.y_margin
        width, height = max_x- min_x, max_y - min_y
        if width > height:
            max_y = min_y + width
        else:
            max_x = min_x + height
        self.turtle.screen.setworldcoordinates(min_x, min_y, max_x, max_y)
        self.turtle.clear()

        self.turtle.color(MapCanvas.OBSTACLE_COLOR)
        for s_x, e_x, s_y, e_y in map_config.obstacles:
            draw_line(self.turtle, (s_x, s_y), (e_x, e_y))

        self.turtle.color(MapCanvas.ROBOT_GOAL_COLOR)
        for rect in map_config.goal_zones:
            draw_rect(self.turtle, rect_points(rect))

        self.turtle.color(MapCanvas.ROBOT_SPAWN_COLOR)
        for rect in map_config.robot_spawn_zones:
            draw_rect(self.turtle, rect_points(rect))

        self.turtle.color(MapCanvas.PED_SPAWN_COLOR)
        for rect in map_config.ped_spawn_zones:
            draw_rect(self.turtle, rect_points(rect))

        self.turtle.color(MapCanvas.ROBOT_ROUTE_COLOR)
        self.turtle.fillcolor(MapCanvas.ROBOT_ROUTE_COLOR)
        for route in map_config.robot_routes:
            for p in route.waypoints:
                draw_circle(self.turtle, p, radius=1)

        self.turtle.color(MapCanvas.TEMP_ROUTE_COLOR)
        self.turtle.fillcolor(MapCanvas.TEMP_ROUTE_COLOR)
        for p in self.temp_waypoints:
            draw_circle(self.turtle, p, radius=1)

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
        self.frame_route_builder = tk.Frame(self.master)

        self.map_canvas = MapCanvas(self.frame_canvas)
        self.text_editor = TextEditor(self.frame_editor)
        self.route_builder = RouteBuilder(
            self.frame_route_builder,
            self.map_canvas.canvas,
            self.map_canvas.render_temp_waypoints,
            lambda: (self.last_config.x_margin, self.last_config.y_margin) \
                if self.last_config else ((0, 1), (0, 1)))

        self.is_shutdown_requested = False
        self.map_rendering_thread: Union[Thread, None] = None
        self.last_text = ''
        self.last_config: Union[VisualizableMapConfig, None] = None
        self._load_example_map()

    def launch(self):
        self.pack()

        def reload_map():
            config_content = self.text_editor.text
            if config_content != self.last_text:
                map_config = parse_mapfile_text(config_content)
                if map_config:
                    self.last_config = map_config
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
