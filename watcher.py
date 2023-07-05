import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
from astro_planner.logger import log


class MyHandler(PatternMatchingEventHandler):
    def __init__(self):
        super(MyHandler, self).__init__(ignore_patterns=["*.ldb"])

    is_changed = True

    def on_created(self, event):
        self.is_changed = True
        log.info(f"Created {event} {self.is_changed}")

    def on_deleted(self, event):
        self.is_changed = True
        log.info(f"Deleted {event} {self.is_changed}")


class Watcher:
    def __init__(
        self,
        directories=["."],
        handler=MyHandler(),
        target=None,
        kwargs=None,
        queue=None,
    ):
        self.observer = Observer()
        self.handler = handler
        self.directories = directories
        self.target = target
        self.kwargs = kwargs
        self.queue = queue

    def update_directories(self):
        for directory in self.directories:
            self.observer.schedule(self.handler, directory, recursive=True)
            log.info("Watcher Running in {}".format(directory))

    def run(self):
        self.update_directories()
        self.observer.start()
        try:
            while True:
                if not self.queue.empty():
                    config = self.queue.get_nowait()
                    if config is None:  # push None to queue, kill wather
                        raise KeyboardInterrupt
                    # env = config.get("env")
                    data_dirs = config.get("directories", {}).get("data_dirs", [])
                    target_dirs = config.get("directories", {}).get("target_dirs", [])
                    calibration_dirs = config.get("directories", {}).get(
                        "calibration_dirs", []
                    )
                    preproc_out_dirs = config.get("directories", {}).get(
                        "preproc_out_dirs", []
                    )

                    if data_dirs is None:
                        data_dirs = []

                    if target_dirs is None:
                        target_dirs = []

                    if calibration_dirs is None:
                        calibration_dirs = []

                    if preproc_out_dirs is None:
                        preproc_out_dirs = []

                    self.directories = (
                        data_dirs + target_dirs + calibration_dirs + preproc_out_dirs
                    )
                    self.observer.stop()
                    self.observer.join()
                    self.observer = Observer()
                    self.update_directories()
                    self.observer.start()
                    self.target.config = config
                    self.handler.is_changed = True
                if self.handler.is_changed:
                    self.handler.is_changed = False
                    log.info("Change detected, triggering update...")
                    if self.target is not None:
                        if self.kwargs is not None:
                            self.target.run(**self.kwargs)
                        else:
                            self.target.run()
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        log.info("\nWatcher Terminated\n")
