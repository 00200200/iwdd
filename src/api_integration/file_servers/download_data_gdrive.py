import gdown
import os
import shutil
from pathlib import Path
import zipfile
import dotenv

def download_drive_folder(folder_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)
    zip_files = list(output_dir.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError("No .zip file found in folder.")

    zip_path = zip_files[0]
    print(f"Found ZIP file: {zip_path}")

    extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    inner_folder = extract_dir / extract_dir.name

    if inner_folder.exists() and inner_folder.is_dir():
        print(f"Flattening nested folder: {inner_folder}")

        for item in inner_folder.iterdir():
            shutil.move(str(item), extract_dir)

        inner_folder.rmdir()

        print("Flattening complete.")

    print(f"Final folder ready at: {extract_dir}\n")
    os.remove(zip_path)

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    download_drive_folder(os.getenv("V_FOLDER_DRIVE_ID"), "data/raw")
    download_drive_folder(os.getenv("L_FOLDER_DRIVE_ID"), "data/raw")