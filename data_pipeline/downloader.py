"""
Streaming Ingestion Engine.

Downloads, caches, and verifies bulk cricket match archives from cricsheet.org.
Uses chunked HTTP streaming to keep memory footprint under 10 MB and includes
checksum/zip integrity verification to prevent corrupted downloads.
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path so 'from data_pipeline...' always resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from data_pipeline.config import RAW_DIR, CRICSHEET_URLS, logger


def verify_zip_archive(zip_path: Path) -> bool:
    """
    Verifies that a file exists, is non-empty, and is a valid, uncorrupted zip archive.

    Args:
        zip_path: Path to the .zip archive on disk.

    Returns:
        True if the zip archive is complete and readable, False otherwise.
    """
    if not zip_path.exists():
        return False

    if zip_path.stat().st_size == 0:
        logger.warning(f"File {zip_path.name} is empty (0 bytes).")
        return False

    if not zipfile.is_zipfile(zip_path):
        logger.warning(f"File {zip_path.name} is not recognized as a valid zip file.")
        return False

    try:
        # testzip() reads headers and verifies CRC-32 checksums of all entries
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupt_file = zf.testzip()
            if corrupt_file is not None:
                logger.error(f"CRC check failed in {zip_path.name} on entry: {corrupt_file}")
                return False
            # Check that it contains match files
            total_files = len(zf.namelist())
            logger.debug(f"Verified {zip_path.name}: contains {total_files} files.")
            return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as err:
        logger.error(f"Zip verification error on {zip_path.name}: {err}")
        return False


def download_file(
    url: str,
    destination: Path,
    chunk_size_bytes: int = 1024 * 1024,  # 1 MB chunk
    force: bool = False
) -> Path:
    """
    Downloads a remote file via chunked HTTP streaming with progress tracking.
    Uses a temporary (.tmp) file to ensure atomic writes—corrupted or aborted
    downloads will never be mistaken for complete files.

    Args:
        url: Remote URL of the file to download.
        destination: Local Path where the final file should be stored.
        chunk_size_bytes: Size of stream chunks (defaults to 1 MB).
        force: If True, re-downloads even if the file is already cached.

    Returns:
        The destination Path to the downloaded and verified file.

    Raises:
        RuntimeError: If download fails or the downloaded file fails verification.
    """
    # 1. Smart Caching: Skip if already downloaded and valid
    if destination.exists() and not force:
        logger.info(f"Checking existing cache for {destination.name}...")
        if verify_zip_archive(destination):
            file_mb = destination.stat().st_size / (1024 * 1024)
            logger.info(
                f"Using cached archive: {destination.name} ({file_mb:.2f} MB). "
                "Skipping download (pass force=True to re-download)."
            )
            return destination
        else:
            logger.warning(f"Cached file {destination.name} is corrupted. Re-downloading...")
            destination.unlink(missing_ok=True)

    # 2. Prepare temporary file path for atomic writing
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    temp_destination.unlink(missing_ok=True)

    logger.info(f"Initiating streaming download from: {url}")
    logger.info(f"Destination: {destination}")

    try:
        # Use stream=True to prevent loading the entire payload into RAM
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            total_size_header = response.headers.get("content-length")
            total_size_bytes = int(total_size_header) if total_size_header else 0
            total_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes else 0

            if total_size_bytes:
                logger.info(f"Total file size: {total_size_mb:.2f} MB")
            else:
                logger.info("Content-Length header not provided by server. Streaming unknown size...")

            bytes_written = 0
            chunks_written = 0
            log_interval = 25  # Log progress every 25 MB

            with open(temp_destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        chunks_written += 1

                        # Log progress at regular intervals
                        if chunks_written % log_interval == 0:
                            mb_done = bytes_written / (1024 * 1024)
                            if total_size_bytes > 0:
                                pct = (bytes_written / total_size_bytes) * 100
                                logger.info(f"Downloaded: {mb_done:.1f} MB / {total_size_mb:.1f} MB ({pct:.1f}%)")
                            else:
                                logger.info(f"Downloaded: {mb_done:.1f} MB...")

        # 3. Atomic rename: move .tmp to final .zip
        temp_destination.replace(destination)
        file_mb = destination.stat().st_size / (1024 * 1024)
        logger.info(f"Download complete: {destination.name} ({file_mb:.2f} MB).")

        # 4. Post-download integrity verification
        logger.info(f"Verifying zip integrity for {destination.name}...")
        if not verify_zip_archive(destination):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded archive {destination.name} failed zip integrity verification.")

        logger.info(f"Verification successful: {destination.name} is ready for parsing.")
        return destination

    except (requests.RequestException, OSError) as exc:
        temp_destination.unlink(missing_ok=True)
        logger.error(f"Download failed for {url}: {exc}")
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_cricsheet(format_type: str = "t20s", force: bool = False) -> Path:
    """
    Downloads and caches a specific Cricsheet match format archive.

    Args:
        format_type: One of 't20s', 'odis', 'tests', or 'all'.
        force: If True, bypasses cache and re-downloads fresh data.

    Returns:
        Path to the verified zip archive in data/raw/.
    """
    key = format_type.lower().strip()
    if key not in CRICSHEET_URLS:
        valid_keys = list(CRICSHEET_URLS.keys())
        raise ValueError(f"Invalid format_type '{format_type}'. Must be one of: {valid_keys}")

    url = CRICSHEET_URLS[key]
    filename = f"{key}_json.zip"
    destination = RAW_DIR / filename

    return download_file(url=url, destination=destination, force=force)


# Quick self-test when running this script directly
if __name__ == "__main__":
    logger.info("Executing Subtask 2 self-test (downloading 't20s' archive)...")
    # Download the t20s archive (~35 MB, contains all official T20Is + 100+ nations)
    zip_path = download_cricsheet("t20s", force=False)
    logger.info(f"Self-test succeeded. Archive available at: {zip_path}")
