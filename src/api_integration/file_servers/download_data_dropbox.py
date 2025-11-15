import requests
import dotenv
import os
import zipfile

def download_dropbox_folder(folder_path, local_zip_path):
    url = "https://content.dropboxapi.com/2/files/download_zip"

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Dropbox-API-Arg": f'{{"path": "{folder_path}"}}'
    }

    print(f"Requesting folder '{folder_path}' from Dropbox")

    response = requests.post(url, headers=headers, stream=True)

    if response.status_code != 200:
        print("Error:", response.text)
        raise Exception(f"Failed with status code {response.status_code}")

    print("Download started")

    with open(local_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Folder downloaded and saved as: {local_zip_path}")


# ------------------ RUN ------------------
if __name__ == "__main__":
    dotenv.load_dotenv()

    vzip_path = os.getenv('LOCAL_VZIP_PATH')
    lzip_path = os.getenv('LOCAL_LZIP_PATH')

    download_dropbox_folder(os.getenv('DROPBOX_VFOLDER_PATH'), vzip_path)
    download_dropbox_folder(os.getenv('DROPBOX_LFOLDER_PATH'), lzip_path)

    with zipfile.ZipFile(vzip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.splitext(vzip_path)[0])

    with zipfile.ZipFile(lzip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.splitext(lzip_path)[0])