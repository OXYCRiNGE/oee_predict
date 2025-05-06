import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging() -> None:
    """
    Настраивает логирование с ротацией файлов и кодировкой UTF-8 в папку logs.
    
    Логи сохраняются в logs/app.log с максимальным размером 10 МБ и 5 архивами.
    Предотвращает дублирование записей, очищая существующие обработчики.
    """
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    # Очистка существующих обработчиков для предотвращения дублирования
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()

    handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 МБ
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)