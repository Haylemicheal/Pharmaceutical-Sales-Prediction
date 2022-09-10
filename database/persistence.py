from sqlalchemy import create_engine
import os
import pymysql

from dotenv import load_dotenv
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"), echo=True)
conn = engine.connect()

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.dataloader import DataLoader
dl = DataLoader()
raw_df = dl.read_csv('../data/store.csv')

raw_df.to_sql('Sales_Data',con = engine)