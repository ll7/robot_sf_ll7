import os
from enum import IntEnum
from time import sleep
from typing import Callable, Union, List, Iterable
from threading import Thread

import tkinter as tk
import tkinter.scrolledtext as tks

from map_editor.map_file_parser import parse_mapfile_text, VisualizableMapConfig
from robot_sf.util.types import Vec2D, Rect


class MapCanvas:
    BG_COLOR = "white"
    OBSTACLE_COLOR = "black"
    ROBOT_GOAL_COLOR = "green"
    ROBOT_SPAWN_COLOR = "blue"
    PED_SPAWN_COLOR = "red"
    PED_GOAL_COLOR = "magenta"
    ROBOT_ROUTE_COLOR = "green"
    PED_CROWDED_COLOR = "orange"
    PED_ROUTE_COLOR = "yellow"

    def __init__(self, frame: tk.Frame):
        self.canvas = tk.Canvas(frame)
        self.width, self.height = 800, 800
        self.canvas.config(width=self.width, height=self.height)
        self.mouse_pos = (0, 0)
        self.last_map_bounds = ((0, 1), (0, 1))

        def track_mouse_pos(event):
            self.mouse_pos = (event.x, event.y)

        def print_map_coords(e):
            x, y = self.mouse_pos
            (min_x, _), (min_y, _) = self.last_map_bounds
            scaling = self.canvas_to_map_scaling()
            map_x, map_y = x * scaling + min_x, y * scaling + min_y
            print(f"[{map_x:.2f}, {map_y:.2f}]")

        self.canvas.bind("<Motion>", track_mouse_pos)
        self.canvas.bind("<Button-1>", print_map_coords)

    def pack(self):
        self.canvas.pack()

    def map_to_canvas_scaling(self) -> float:
        canvas_width, canvas_height = self.width, self.height
        (min_x, max_x), (min_y, max_y) = self.last_map_bounds
        map_width, map_height = max_x - min_x, max_y - min_y
        x_scale, y_scale = canvas_width / map_width, canvas_height / map_height
        return min(x_scale, y_scale)

    def canvas_to_map_scaling(self) -> float:
        return 1 / self.map_to_canvas_scaling()

    def render(self, map_config: VisualizableMapConfig):
        self.last_map_bounds = (map_config.x_margin, map_config.y_margin)
        self.canvas.delete("all")
        self.canvas.configure(bg=MapCanvas.BG_COLOR)
        scaling = self.map_to_canvas_scaling()

        def rect_points(rect: Rect) -> List[Vec2D]:
            def add_vec(v1, v2):
                return (v1[0] + v2[0], v1[1] + v2[1])

            def sub_vec(v1, v2):
                return (v1[0] - v2[0], v1[1] - v2[1])

            p1, p2, p3 = rect
            p4 = add_vec(sub_vec(p3, p2), p1)
            return [p1, p2, p3, p4]

        def scale(p: Vec2D) -> Vec2D:
            (min_x, _), (min_y, _) = map_config.x_margin, map_config.y_margin
            return (p[0] - min_x) * scaling, (p[1] - min_y) * scaling

        def draw_zone(zone_id: int, rect: List[Vec2D], color="black"):
            p1, p2, p3, p4 = rect
            self.canvas.create_line(scale(p1), scale(p2), fill=color)
            self.canvas.create_line(scale(p2), scale(p3), fill=color)
            self.canvas.create_line(scale(p3), scale(p4), fill=color)
            self.canvas.create_line(scale(p4), scale(p1), fill=color)

            min_x, max_x = min(p1[0], p2[0], p3[0], p4[0]), max(p1[0], p2[0], p3[0], p4[0])
            min_y, max_y = min(p1[1], p2[1], p3[1], p4[1]), max(p1[1], p2[1], p3[1], p4[1])
            middle = min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2
            middle = scale(middle)
            self.canvas.create_text(
                middle[0],
                middle[1],
                text=str(zone_id),
                fill="black",
                font=("Helvetica 10 bold"),
            )

        def draw_waypoint(p: Vec2D, r: float, color="black", fill=None):
            (x, y), r = scale(p), r * scaling
            fill = fill if fill else color
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, fill=fill)

        for s_x, e_x, s_y, e_y in map_config.obstacles:
            if (s_x, s_y) != (e_x, e_y):
                self.canvas.create_line(
                    scale((s_x, s_y)), scale((e_x, e_y)), fill=MapCanvas.OBSTACLE_COLOR
                )

        for i, rect in enumerate(map_config.robot_spawn_zones):
            draw_zone(i, rect_points(rect), MapCanvas.ROBOT_SPAWN_COLOR)

        for i, rect in enumerate(map_config.robot_goal_zones):
            draw_zone(i, rect_points(rect), MapCanvas.ROBOT_GOAL_COLOR)

        for i, rect in enumerate(map_config.ped_spawn_zones):
            draw_zone(i, rect_points(rect), MapCanvas.PED_SPAWN_COLOR)

        for i, rect in enumerate(map_config.ped_goal_zones):
            draw_zone(i, rect_points(rect), MapCanvas.PED_GOAL_COLOR)

        for i, rect in enumerate(map_config.ped_crowded_zones):
            draw_zone(i, rect_points(rect), MapCanvas.PED_CROWDED_COLOR)

        for route in map_config.robot_routes:
            for p in route.waypoints:
                draw_waypoint(p, 1, MapCanvas.ROBOT_ROUTE_COLOR)

        for route in map_config.ped_routes:
            for p in route.waypoints:
                draw_waypoint(p, 1, MapCanvas.PED_ROUTE_COLOR)


class MapToolbarMode(IntEnum):
    NONE = 0
    NEW_ROBOT_SPAWN = 1
    NEW_ROBOT_GOAL = 2
    NEW_ROBOT_ROUTE = 3
    NEW_PED_SPAWN = 4
    NEW_PED_GOAL = 5
    NEW_PED_ROUTE = 6
    NEW_OBSTACLE = 7


class MapEditorToolbar:
    COLOR_BTN_DEFAULT = "gray"
    COLOR_BTN_HIGHLIGHTED = "yellow"

    def __init__(self, frame: tk.Frame, on_mode_changed: Callable[[MapToolbarMode], None]):
        self.on_mode_changed = on_mode_changed
        self.mode = MapToolbarMode.NONE

        self.buttons_by_mode = {}
        self.buttons_by_mode[MapToolbarMode.NEW_ROBOT_SPAWN] = tk.Button(
            frame,
            text="Robot Spawn",
            command=lambda: self.change_mode(MapToolbarMode.NEW_ROBOT_SPAWN),
        )
        self.buttons_by_mode[MapToolbarMode.NEW_ROBOT_GOAL] = tk.Button(
            frame,
            text="Robot Goal",
            command=lambda: self.change_mode(MapToolbarMode.NEW_ROBOT_GOAL),
        )
        self.buttons_by_mode[MapToolbarMode.NEW_ROBOT_ROUTE] = tk.Button(
            frame,
            text="Robot Route",
            command=lambda: self.change_mode(MapToolbarMode.NEW_ROBOT_ROUTE),
        )
        self.buttons_by_mode[MapToolbarMode.NEW_PED_SPAWN] = tk.Button(
            frame,
            text="Ped Spawn",
            command=lambda: self.change_mode(MapToolbarMode.NEW_PED_SPAWN),
        )
        self.buttons_by_mode[MapToolbarMode.NEW_PED_GOAL] = tk.Button(
            frame,
            text="Ped Goal",
            command=lambda: self.change_mode(MapToolbarMode.NEW_PED_GOAL),
        )
        self.buttons_by_mode[MapToolbarMode.NEW_PED_ROUTE] = tk.Button(
            frame,
            text="Ped Route",
            command=lambda: self.change_mode(MapToolbarMode.NEW_PED_ROUTE),
        )

    @property
    def buttons(self) -> Iterable[tk.Button]:
        return self.buttons_by_mode.values()

    def pack(self):
        for btn in self.buttons:
            btn.pack(side=tk.TOP, fill="x")
            btn.configure(background=MapEditorToolbar.COLOR_BTN_DEFAULT)

    def change_mode(self, new_mode: MapToolbarMode):
        if self.mode == new_mode:
            self.mode = MapToolbarMode.NONE
        else:
            self.mode = new_mode

        for mode, btn in self.buttons_by_mode.items():
            highlight = mode == self.mode
            new_bg = (
                MapEditorToolbar.COLOR_BTN_HIGHLIGHTED
                if highlight
                else MapEditorToolbar.COLOR_BTN_DEFAULT
            )
            btn.configure(background=new_bg)

        self.on_mode_changed(self.mode)


class TextEditor:
    # TODO: support undo / redo logic with ctrl+z / ctrl+y

    def __init__(self, frame: tk.Frame, clipboard: Callable[[], str]):
        self.input = tks.ScrolledText(frame)
        self.input.bind("<Control-Key-a>", lambda e: self.select_all())
        # self.input.bind("<Control-Key-v>", lambda e: self.insert_text(clipboard()))

    @property
    def text(self) -> str:
        return self.input.get("1.0", "end-1c")

    @property
    def has_selected_text(self) -> bool:
        return not self.input.selection_get()

    def pack(self):
        self.input.pack(side=tk.LEFT, fill="both")

    def clear_text(self):
        self.input.delete("1.0", tk.END)

    def append_text(self, text: str):
        self.input.insert(tk.END, text)

    def select_all(self):
        self.input.tag_add(tk.SEL, "1.0", tk.END)
        self.input.mark_set(tk.INSERT, "1.0")
        self.input.see(tk.INSERT)
        return "break"

    # def insert_text(self, text: str):
    #     if self.has_selected_text:
    #         self.input.delete(tk.SEL_FIRST, tk.SEL_LAST)
    #     self.input.insert(tk.CURRENT, text)
    #     return "break"


class MapEditor:
    def __init__(self):
        TITLE = "RobotSF Map Editor"
        self.master = tk.Tk()
        self.master.resizable(False, False)
        self.master.title(TITLE)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame_editor = tk.Frame(self.master)
        self.frame_toolbar = tk.Frame(self.master)
        self.frame_canvas = tk.Frame(self.master)

        self.map_canvas = MapCanvas(self.frame_canvas)
        self.text_editor = TextEditor(self.frame_editor, self.master.clipboard_get)
        self.map_toolbar = MapEditorToolbar(self.frame_toolbar, lambda mode: None)

        self.is_shutdown_requested = False
        self.map_rendering_thread: Union[Thread, None] = None
        self.last_text = ""
        self.last_config: Union[VisualizableMapConfig, None] = None
        self._load_example_map()

    def launch(self):
        self.pack()

        def reload_map():
            config_content = self.text_editor.text
            map_config = (
                parse_mapfile_text(config_content) if config_content != self.last_text else None
            )
            self.last_config = map_config if map_config else self.last_config
            self.last_text = config_content if map_config else self.last_text
            if map_config:
                try:
                    self.map_canvas.render(map_config)
                except BaseException:
                    print("unable to draw map")

        def reload_map_as_daemon(frequency_hz: float, is_term: Callable[[], bool]):
            reload_intercal_secs = 1 / frequency_hz
            while not is_term():
                reload_map()
                sleep(reload_intercal_secs)

        RELOAD_FREQUENCY = 5
        args = (RELOAD_FREQUENCY, lambda: self.is_shutdown_requested)
        self.map_rendering_thread = Thread(target=reload_map_as_daemon, args=args)
        self.map_rendering_thread.start()
        self.master.mainloop()

    def pack(self):
        self.frame_toolbar.pack(side=tk.RIGHT, fill="y")
        self.frame_canvas.pack(side=tk.RIGHT)
        self.frame_editor.pack(side=tk.LEFT, fill="both")
        self.map_canvas.pack()
        self.text_editor.pack()
        self.map_toolbar.pack()

    def on_closing(self):
        if self.map_rendering_thread:
            self.is_shutdown_requested = True
            self.map_rendering_thread.join()
        self.master.destroy()

    def _load_example_map(self):
        current_dir = os.path.dirname(__file__)
        example_filepath = os.path.join(current_dir, "map_example.json")
        with open(example_filepath, "r") as file:
            text = file.read()
        self.text_editor.clear_text()
        self.text_editor.append_text(text)
