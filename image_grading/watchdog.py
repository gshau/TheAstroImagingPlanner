import time
import logging
from preprocessing import update_db_with_matching_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info(f"Starting watchdog on data directory")
    while True:
        update_db_with_matching_files()
        time.sleep(5)
