# Документация API прогнозирования OEE

## Общие сведения

**Название API**: API прогноза OEE  
**Описание**: API предназначен для генерации прогнозов показателя общей эффективности оборудования (OEE) на основе исторических данных.  
**Версия**: 1.0  
**Формат ответа**: JSON

API предоставляет единственную конечную точку для получения прогнозов OEE на заданное количество дней (от 1 до 60).

---

## Конечные точки API

### 1. Получение прогноза OEE

**Метод**: GET  
**Путь**: `/api/get/forecast`  
**Описание**: Генерирует прогноз OEE на указанное количество дней.  
**Параметры запроса**:
- `days` (integer): Количество дней для прогноза (от 1 до 60). По умолчанию: 60.

**Пример запроса**:
```
http://{host}:{port}/api/get/forecast?days=30
```

**Успешный ответ**:
- **Код состояния**: 200 OK
- **Формат ответа**:
```json
{
  "forecast": [
    {
      "date": "2023-10-01T00:00:00",
      "oee": 0.85
    },
    ...
  ]
}
```
- `forecast`: Массив объектов, содержащих прогнозы OEE. Количество объектов равно значению параметра `days`.
  - `date`: Дата прогноза в формате ISO 8601 (`YYYY-MM-DDTHH:MM:SS`). Указывает день, для которого сделан прогноз.
  - `oee`: Предсказанное значение OEE (число с плавающей точкой) для соответствующего дня, обычно в диапазоне от 0 до 1.

**Ошибки**:
- **400 Bad Request**:
  - Если `days` меньше 1: `"Количество дней должно быть не менее 1"`
  - Если `days` больше 60: `"Количество дней не должно превышать 60"`
- **500 Internal Server Error**:
  - Если файл данных или модель не найдены: `"Файл не найден: <имя_файла>"`
  - Если загруженный объект не является моделью: `"Загруженный объект не является моделью"`
  - Для прочих ошибок сервера: `"Ошибка сервера: <описание_ошибки>"`

**Пример ответа с ошибкой**:
```json
{
  "detail": "Количество дней должно быть не менее 1"
}
```

---

## Установка и запуск

**Установка с помощью Docker**:
```bash
docker-compose up --build
```

---

## Обработка данных

1. **Загрузка данных**:
   - Данные загружаются из Excel-файла, указанного в переменной окружения `DATA_FILE` (по умолчанию: `dataset\signal_data_full.xlsx`).
   - Данные обрабатываются функцией `preprocess_data` для создания тренировочных и тестовых наборов.

2. **Модель**:
   - Если файл модели (указан в `MODEL_FILE`, по умолчанию: `model\best_pipeline_model.pkl`) существует, загружается сохраненная модель.
   - Если модель отсутствует, выполняется обучение новой модели с использованием функции `train_model`.

3. **Прогнозирование**:
   - Прогноз генерируется функцией `forecast_future` на основе тестовой выборки и указанного горизонта прогнозирования (`days`).
   - Результат возвращается в формате JSON.

---

## Логирование

- Все действия API (запуск, запросы, ошибки) логируются в файл `logs/app.log`.
- Логи хранятся с ротацией (максимум 10 МБ на файл, до 5 архивов).
- Формат логов настраивается через функцию `setup_logging` из модуля `logging_config`.

---

## Ограничения

- Максимальный горизонт прогнозирования: 60 дней.
- API не поддерживает изменение данных или модели через запросы.
- В случае отсутствия файла данных или модели API вернет ошибку 500.
