import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv
import os
import urllib3
import pandas as pd
import urllib.parse
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import logging
from logging_config import setup_logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Настройка логирования
setup_logging()

def load_credentials():
    """Загружает логин и пароль из .env файла."""
    load_dotenv()
    login = os.getenv('USER_LOGIN')
    password = os.getenv('USER_PASSWORD')
    winnum_url = os.getenv('WINNUM_URL')
    if not login or not password:
        raise ValueError("Логин или пароль не найдены в .env файле")
    return login, password, winnum_url

def authorize(login, password, winnum_url):
    """Авторизация на сервере и возвращение сессии и URL."""
    session = requests.Session()
    login_url = f"{winnum_url}/Winnum/servlets/WinnumLogin"
    payload = {
        'uid': login,
        'pwd': password,
        'smbt': 'Войти'
    }
    login_response = session.post(login_url, data=payload, verify=False)
    if login_response.ok:
        logging.info("Авторизация выполнена успешно")
        url = f"{winnum_url}/Winnum/views/pages/app/agw.jsp"
        return session, url
    else:
        raise ConnectionError("Ошибка авторизации")

def execute_api(session, url, params, method='GET'):
    """Выполнение API-запроса с исправлением URL."""
    req = requests.Request(method, url, params=params)
    prepared = req.prepare()
    prepared.url = prepared.url.replace('%3A', ':')
    r = session.request(method=prepared.method, url=prepared.url, cookies=session.cookies, verify=False)
    return r

def get_items(response, parser='html.parser', tag='item', show_xml=False):
    """Извлечение элементов из ответа API."""
    soup = BeautifulSoup(response.content, parser)
    if show_xml:
        xml = soup.prettify()
    items = soup.find_all(tag)
    return items

def process_signal_data(df_uuid, df_tags, start_date='2021-01-01', end_date=None, output_file='signal_data_full.xlsx'):
    """
    Обработка данных сигналов и сохранение в Excel в папку dataset с прогресс-баром.
    
    Параметры:
    - df_uuid (pd.DataFrame): DataFrame с UUID и именами машин.
    - df_tags (pd.DataFrame): DataFrame с тегами и их именами.
    - start_date (str): Начальная дата в формате 'YYYY-MM-DD' (по умолчанию '2021-01-01').
    - end_date (str): Конечная дата в формате 'YYYY-MM-DD' (по умолчанию вчера).
    - output_file (str): Имя файла для сохранения результата (будет сохранён в папке dataset).
    """
    if end_date is None:
        end_date = (datetime.today().date() - timedelta(days=1)).isoformat()

    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, output_file)

    login, password, winnum_url = load_credentials()
    session, url = authorize(login, password, winnum_url)

    try:
        all_data = []
        for _, uuid_row in tqdm(df_uuid.iterrows(), total=df_uuid.shape[0], desc="Обработка UUID"):
            uuid = uuid_row['productuuid']
            machine_name = uuid_row['decoded_name']
            uuid_data = {}

            for _, tag_row in tqdm(df_tags.iterrows(), total=df_tags.shape[0], desc=f"Теги для {machine_name}", leave=False):
                tag_id = tag_row['WNTag']
                tag_name = tag_row['name']
                params = {
                    'rpc': 'winnum.views.url.WNConnectorHelper',
                    'men': 'getSignal',
                    'uuid': uuid,
                    'signal': f'121.{tag_id}.SCHED_PRIO.HOUR',
                    'stype': 'bytime',
                    'order': 'asc',
                    'start': f'{start_date} 00:00:00',
                    'end': f'{end_date} 23:59:59',
                    'mode': 'yes'
                }
                r = execute_api(session, url, params)
                items = get_items(r, show_xml=False)

                for item in items:
                    event_time = urllib.parse.unquote(item['event_time'])
                    date = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')
                    if date not in uuid_data:
                        uuid_data[date] = {'Объект': machine_name, 'Дата': date}
                    uuid_data[date][tag_name] = float(float(item['value']) / 3600000)

            all_data.extend(uuid_data.values())

        df = pd.DataFrame(all_data)
        tag_columns = ['Объект', 'Дата'] + list(df_tags['name'])
        for col in tag_columns:
            if col not in df.columns:
                df[col] = None
        df = df[tag_columns]
        df.to_excel(output_path, index=False)
        logging.info(f"Данные сохранены в {output_path}")
    finally:
        session.close()