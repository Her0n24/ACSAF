"""
This script downloads the latest AIFS run from ECMWF using wget to input folder.

format: wget [ROOT]/20240301/00z/aifs-single/0p25/oper/20240301060000-24h-oper-fc.grib2

Where ROOT is ECMWF:  https://data.ecmwf.int/forecasts

"""
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from http.client import IncompleteRead
from urllib3.exceptions import ProtocolError
import os
import logging
import time
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
                    )

# Define run to 2 digits
run = "00"
run = run.zfill(2)

# Define a function that use reponse to donwload the file from a given url
def download_file(url, local_filename, max_retries: int = 4, backoff_seconds: int = 5) -> None:
    if os.path.exists(local_filename):
        logging.info(f"File {local_filename} already exists. Skipping download.")
        return

    # Configure a requests Session with urllib3 Retry for connection-level failures
    session = requests.Session()
    retries = Retry(
        total=0,  # we'll handle body-read retries explicitly below
        backoff_factor=backoff_seconds,
        status_forcelist=(500, 502, 503, 504),  # Remove 429 from here as we handle it separately
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    tmp_name = local_filename + ".part"
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            with session.get(url, stream=True, timeout=(10, 120)) as r:
                # Handle rate limiting (429) and other HTTP errors with retry logic
                if r.status_code == 429:
                    if attempt > max_retries:
                        logging.error(f"Rate limited (429) for {url}. Exceeded max retries ({max_retries}).")
                        return
                    # Exponential backoff for rate limiting, starting with longer delays
                    sleep_time = backoff_seconds * (2 ** (attempt - 1)) + (10 * attempt)  # Extra delay for rate limits
                    logging.warning(f"Rate limited (429) for {url}. Retrying in {sleep_time} seconds... (Attempt {attempt}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                elif r.status_code != 200:
                    if r.status_code in [500, 502, 503, 504] and attempt <= max_retries:
                        # Server errors - retry with shorter delay
                        sleep_time = backoff_seconds * (2 ** (attempt - 1))
                        logging.warning(f"Server error {r.status_code} for {url}. Retrying in {sleep_time} seconds... (Attempt {attempt}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logging.error(f"Failed to download: {url} - Status code: {r.status_code}")
                        return

                total_size = int(r.headers.get('content-length', 0))
                try:
                    # Disable tqdm progress bar when not in interactive terminal (e.g., cron)
                    with open(tmp_name, 'wb') as f, tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        disable=not sys.stdout.isatty(),
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                bar.update(len(chunk))
                    os.replace(tmp_name, local_filename)
                    logging.info(f"Downloaded: {local_filename}")
                    return
                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, OSError, IncompleteRead, ProtocolError) as e:
                    # Clean up partial file on error and retry (if attempts remain)
                    logging.warning(
                        f"Download interrupted for {url}: {e}. Removing partial file. Attempt {attempt}/{max_retries}.")
                    try:
                        if os.path.exists(tmp_name):
                            os.remove(tmp_name)
                    except Exception:
                        pass
                    if attempt > max_retries:
                        logging.error(f"Exceeded retry attempts for {url}: {e}")
                        return
                    sleep_time = backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_time)
                    continue
        except requests.exceptions.ConnectionError as ex:
            if attempt > max_retries:
                logging.error(f"Failed to download {url}: {ex}")
                return
            sleep_time = backoff_seconds * (2 ** (attempt - 1))
            logging.warning(f"Connection error for {url}: {ex}. Retrying in {sleep_time} seconds... (Attempt {attempt}/{max_retries})")
            time.sleep(sleep_time)
            continue
        except Exception as ex:
            logging.error(f"Failed to download {url}: {ex}")
            return

# url = f"https://data.ecmwf.int/forecasts/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-18h-oper-fc.grib2"
# download_file(url, f"input/{today_str}{run}0000-18h-oper-fc.grib2")

# url = f"https://data.ecmwf.int/forecasts/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-42h-oper-fc.grib2"   
# download_file(url, f"input/{today_str}{run}0000-42h-oper-fc.grib2")

