import logging
import logging.handlers
import os
from .globals import DATA_DIR


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(module)s:%(lineno)d %(message)s",
)
log = logging.getLogger(__name__)

handler = logging.handlers.WatchedFileHandler(
    f"{DATA_DIR}/data/logs/planner.log", mode="w"
)

formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
handler.setFormatter(formatter)
root = logging.getLogger()

root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

logger = logging.getLogger("werkzeug")
logger.handlers.clear()
logger.setLevel(logging.ERROR)
logger.propagate = False
