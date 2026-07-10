# INSTALL.md — Установка и развёртывание

> Пошаговая инструкция для развёртывания проекта RecursiveMAS с нуля.

---

## Шаг 1 — Клонирование репозитория

```bash
cd ~/workspace/projects/recursivemas/experiments
git clone https://github.com/RecursiveMAS/RecursiveMAS.git original
```

## Шаг 2 — Создание виртуального окружения

```bash
# Создаём venv с CUDA поддержкой
python3 -m venv ~/workspace/venvs/recursivemas/cuda_12_x
source ~/workspace/venvs/recursivemas/cuda_12_x/bin/activate

# Обновляем pip
pip install --upgrade pip
```

## Шаг 3 — Установка зависимостей

```bash
cd ~/workspace/projects/recursivemas/experiments/original
pip install -r requirements.txt
```

Если какие-то пакеты не устанавливаются из-за проблем с сетью:
```bash
pip install --retries 5 --timeout 30 <package_name>
```

## Шаг 4 — Проверка CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB')"
```

Ожидаемый вывод: CUDA доступна, RTX 4060 Ti 16GB (или другая GPU).

## Шаг 5 — Проверка загрузки моделей

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Transformers OK')
# Попробуй загрузить маленькую модель
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)
print('Tokenizer loaded OK')
"
```

## Шаг 6 — Создание .venv скрипта

Создай файл `.venv` в корневой папке проекта:

```bash
#!/usr/bin/env bash
# Активирует виртуальное окружение recursivemas
#
# ИСПОЛЬЗОВАНИЕ:
#   source .venv

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Ошибка: запускай через source:  source .venv"
    exit 1
fi

source ~/workspace/venvs/recursivemas/cuda_12_x/bin/activate
```

```bash
chmod +x .venv
```

## Шаг 7 — Проверка работы inference

```bash
source .venv
cd experiments/original

# Быстрый тест с Sequential Light на небольшом числе сэмплов
python inference/run.py \
  --style sequential_light \
  --dataset math500 \
  --device cuda \
  --num_samples 5
```

Если вывел результаты — всё работает! ✅

## Шаг 8 — Опционально: HuggingFace авторизация

Для загрузки gated моделей (Llama и др.):

```bash
pip install huggingface_hub
huggingface-cli login
# Вставь токен с https://huggingface.co/settings/tokens
```

---

## Повторное развёртывание

Если нужно развернуть с нуля:
1. Удали venv: `rm -rf ~/workspace/venvs/recursivemas/cuda_12_x`
2. Начни с Шага 1

## Troubleshooting

### Ошибка: CUDA out of memory
- Используй Sequential Light (~4.2B) вместо Scaled (~11B)
- Уменьши batch size
- Используй `--device cuda:1` для RTX 4060 Ti

### Ошибка: модель не загружается
- Проверь интернет-соединение
- Попробуй `huggingface-cli download <model>` вручную
- Проверь авторизацию HF для gated моделей

### Ошибка: ImportError при запуске
- Пересоздай venv с нуля (Шаг 2)
- Проверь, что все зависимости из requirements.txt установлены
