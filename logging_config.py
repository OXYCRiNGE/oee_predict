import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging() -> None:
    """
    Настраивает логирование с ротацией файлов в папку logs и выводом в консоль.
    
    Логи сохраняются в logs/app.log с максимальным размером 10 МБ и 5 архивами.
    Те же сообщения логов выводятся в консоль.
    Предотвращает дублирование записей, очищая существующие обработчики.
    """
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    # Очистка существующих обработчиков для предотвращения дублирования
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()

    # Форматтер для логов
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 МБ
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Добавление обработчиков к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)