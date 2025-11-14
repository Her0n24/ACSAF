import os 
import time
import logging
logging.basicConfig(
    level=logging.INFO,
    datefmt= '%Y-%m-%d %H:%M:%S',
                    )

def rm_old_files(folder_paths: list[str], age_threshold_day: int = 2) -> None:
    logging.info("Starting scheduled file cleanup...")

    age_threshold = age_threshold_day * 24 * 60 * 60  # 2 days in seconds
    now = time.time()
    logging.info(f"Deletion age threshold set to {age_threshold/(24 * 60 * 60)} days")

    for folder_path in folder_paths:
        for files in os.listdir(folder_path):
            file_path = os.path.join(folder_path, files)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > age_threshold:
                    os.remove(file_path)
                    logging.debug(f"Deleted: {file_path}")
            else:
                logging.info(f"Skipped (not a file): {file_path}")

if __name__ == "__main__":
    pass