import logging
import logging.handlers

import os


handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "/logs/planner.log"), mode="w"
)

formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(module)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()

root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(module)s %(message)s",
)

log = logging.getLogger(__name__)
