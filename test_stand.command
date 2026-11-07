#!/bin/zsh
set -u

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
TEST_DATA_DIR="${CV_TEST_DATA_DIR:-/Users/gregorynovikov/Downloads}"
ASSET_DIR="$PROJECT_DIR/data/raw/test-stand"
PROJECT_FILE="$PROJECT_DIR/configs/test_stand.local.json"
TEMPLATE_FILE="$PROJECT_DIR/configs/test_stand.example.json"
RUN_DIR="$PROJECT_DIR/runs/test-stand"

fail() {
  echo "Ошибка: $1"
  read -r "?Нажмите Enter, чтобы закрыть окно."
  exit 1
}

[[ -x "$PROJECT_DIR/.venv/bin/python" ]] || fail "не найдено окружение .venv. Сначала запустите setup.command."
mkdir -p "$ASSET_DIR" "$RUN_DIR" || fail "не удалось создать локальные папки стенда."

for index in 1 2 3 4 5; do
  source_file="$TEST_DATA_DIR/camera_new_${index}.mkv"
  target_file="$ASSET_DIR/camera_new_${index}.mkv"
  [[ -f "$source_file" ]] || fail "не найден файл $source_file"
  if [[ -e "$target_file" && ! -L "$target_file" ]]; then
    fail "$target_file уже существует и не является ссылкой."
  fi
  if [[ ! -L "$target_file" ]]; then
    ln -s "$source_file" "$target_file" || fail "не удалось подключить camera_new_${index}.mkv"
  fi
done

for image_name in office.jpg office_camera.png; do
  [[ -f "$TEST_DATA_DIR/$image_name" ]] || fail "не найден файл $TEST_DATA_DIR/$image_name"
  cp "$TEST_DATA_DIR/$image_name" "$ASSET_DIR/$image_name" || fail "не удалось скопировать $image_name"
done

if [[ ! -f "$PROJECT_FILE" ]]; then
  cp "$TEMPLATE_FILE" "$PROJECT_FILE" || fail "не удалось создать конфигурацию стенда."
fi

cd "$PROJECT_DIR" || fail "не удалось открыть каталог проекта."
echo "Тестовый стенд: http://127.0.0.1:8766"
echo "Проект: $PROJECT_FILE"
echo "Результаты: $RUN_DIR"
"$PROJECT_DIR/.venv/bin/python" -m src.app \
  --port 8766 \
  --project "$PROJECT_FILE" \
  --runs "$RUN_DIR" \
  --open
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  fail "стенд завершился с кодом $STATUS. Проверьте, не занят ли порт 8766."
fi
exit 0
