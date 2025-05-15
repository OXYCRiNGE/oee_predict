from environs import Env
import os

env = Env()
env.read_env()


DATA_FILE=env.path('DATA_FILE')
MODEL_FILE=env.path('MODEL_FILE')
USER_LOGIN=env.str('USER_LOGIN')
USER_PASSWORD=env.str('USER_PASSWORD')
WINNUM_URL=env.str('WINNUM_URL')
START_WITH_DOWNLOAD=env.bool('START_WITH_DOWNLOAD', True)
DB_HOST=env.str("DB_HOST")
DB_PORT=env.str("DB_PORT")
DB_USER=env.str("DB_USER")
DB_PASSWORD=env.str("DB_PASSWORD")
DB_NAME=env.str("DB_NAME")