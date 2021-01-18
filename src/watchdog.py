import time
import logging
from preprocessing import update_db_with_matching_files, update_db_with_targets


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info(f"Starting watchdog on data directory")
    while True:
        file_list = None
        update_db_with_matching_files(file_list=file_list)
        update_db_with_targets(file_list=None)
        time.sleep(60)
