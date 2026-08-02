"""Helper for pulling datasets from Kaggle via the official API."""

from pathlib import Path

from dotenv import load_dotenv


def download_kaggle_dataset(dataset: str, dest_dir: str | Path, unzip: bool = True) -> Path:
    """Download a Kaggle dataset into ``dest_dir`` using the authenticated Kaggle API.

    Args:
        dataset: Kaggle dataset slug in "owner/dataset-name" form, as shown in the
            dataset's URL (e.g. "zynicide/wine-reviews").
        dest_dir: Directory to download (and optionally unzip) the dataset into.
            Created if it doesn't exist.
        unzip: Whether to unzip downloaded files in place.

    Returns:
        The destination directory as a Path.

    Requires KAGGLE_USERNAME and KAGGLE_KEY, either set in the environment or in a
    .env file at the project root (see .env.example), or a token at
    ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api#authentication).
    """
    load_dotenv()

    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=unzip)

    return dest_dir


def download_kaggle_competition(
    competition: str, dest_dir: str | Path, unzip: bool = True, file_name: str | None = None
) -> Path:
    """Download competition data into ``dest_dir`` using the authenticated Kaggle API.

    Args:
        competition: Competition slug as shown in its URL (e.g.
            "ga-customer-revenue-prediction"). You must have accepted the
            competition's rules on kaggle.com first, or the download will 403.
        dest_dir: Directory to download (and optionally unzip) files into. Created
            if it doesn't exist.
        unzip: Whether to unzip downloaded files in place.
        file_name: If set, download only this file instead of the full archive.

    Returns:
        The destination directory as a Path.

    Requires KAGGLE_USERNAME and KAGGLE_KEY, either set in the environment or in a
    .env file at the project root (see .env.example), or a token at
    ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api#authentication).
    """
    load_dotenv()

    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    if file_name:
        api.competition_download_file(competition, file_name, path=str(dest_dir))
    else:
        api.competition_download_files(competition, path=str(dest_dir))
    if unzip:
        for zip_path in dest_dir.glob("*.zip"):
            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            zip_path.unlink()

    return dest_dir
