from pathlib import Path
from rembg import remove, new_session
from tqdm import tqdm
import os

# ================== НАСТРОЙКИ ==================
# Укажи папку с персонажами
ROOT_DIR = Path("/Users/jurasik/Downloads/characters")

# Какие расширения обрабатывать
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ==============================================

def process_folder(folder: Path, session):
    files = [f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS and f.is_file()]

    if not files:
        return

    print(f"\nОбрабатываю: {folder.name} ({len(files)} файлов)")

    for file in tqdm(files, desc=folder.name, leave=False):
        try:
            # Читаем
            with open(file, "rb") as f:
                input_data = f.read()

            # Удаляем фон
            output_data = remove(input_data, session=session)

            # Сохраняем обратно в ту же папку как PNG (чтобы сохранить прозрачность)
            output_path = file.with_suffix(".png")
            with open(output_path, "wb") as f:
                f.write(output_data)

            # Если исходник был не png — удаляем старый файл
            if file.suffix.lower() != ".png":
                file.unlink()

        except Exception as e:
            print(f"Ошибка с файлом {file.name}: {e}")


def main():
    session = new_session()  # загружаем модель один раз

    # Если внутри ROOT_DIR лежат папки персонажей
    character_folders = [d for d in ROOT_DIR.iterdir() if d.is_dir()]

    if character_folders:
        for folder in character_folders:
            process_folder(folder, session)
    else:
        # Если указана сразу папка с фотками
        process_folder(ROOT_DIR, session)

    print("\nГотово! Фон удалён, файлы сохранены в те же папки.")


if __name__ == "__main__":
    main()