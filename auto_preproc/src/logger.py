# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(module)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
# )
# log = logging.getLogger()

from astro_planner.logger import log


# import logging
# import logging.handlers
# import os
# from astro_planner.globals import BASE_DIR


# logging.basicConfig(
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     format="%(asctime)s %(levelname)s %(module)s:%(lineno)d %(message)s",
# )
# log = logging.getLogger(__name__)

# handler = logging.handlers.WatchedFileHandler(
#     f"{BASE_DIR}/data/logs/autopreproc.log", mode="w"
# )

# formatter = logging.Formatter(
#     fmt="%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
# )
# handler.setFormatter(formatter)
# root = logging.getLogger()

# root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
# root.addHandler(handler)

# logger = logging.getLogger("werkzeug")
# logger.handlers.clear()
# logger.setLevel(logging.ERROR)
# logger.propagate = False
