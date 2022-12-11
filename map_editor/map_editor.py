from time import sleep
from typing import Tuple, Callable, Union
from threading import Thread

import turtle
import tkinter as tk
import tkinter.scrolledtext as tks

from map_editor.map_file_parser import \
    parse_mapfile_text, VisualizableMapConfig


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

    def launch(self):
        def reload_map():
            config_content = self.editor_text_input.get("1.0", 'end-1c')
            if config_content != self.last_text:
                map_config = parse_mapfile_text(config_content)
                if map_config:
                    self.last_text = config_content
                    self._render_map_canvas(map_config)
                else:
                    print('parsing config file failed!')
                    # TODO: think of showing some hint that there was a parser error
                    #       any maybe augment in the text view where the error occured
                    pass

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
        for s_x, e_x, s_y, e_y in map_config.obstacles:
            self.my_turtle.up()
            self.my_turtle.setpos(s_x, s_y)
            self.my_turtle.down()
            self.my_turtle.setpos(e_x, e_y)
        # TODO: hide turtle cursor
