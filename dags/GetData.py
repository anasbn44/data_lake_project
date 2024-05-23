import pandas as pd
import json
import socket
import time
import logging
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.bash import BashOperator
from hdfs import InsecureClient
from airflow.models import Connection
from airflow.utils import db
from io import StringIO
import uuid
from datetime import timedelta
from get_data.tableExtract import get_tables
from get_data.budgetEco import get_pdf_content, extract_table, extract_text
import asyncio
import os
from pdf2image import convert_from_bytes
import get_data.ocr as ocr


default_args = {
    'owner': 'bcp',
    'start_date': datetime(2024, 5, 1, 22, 5)
}

def get_test_data(file_path):
    df = pd.read_csv(file_path, delimiter=',', quotechar='"')
    print(df.head())
    df.to_csv('/tmp/test_data.csv',index=False)

def get_pdf():
    output_dir = '/tmp/raw_pdf/'
    save_path = '/tmp/raw_pdf/Budget_économique_prévisionnel_2024.pdf'
    pdf_content = asyncio.run(get_pdf_content())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        file.write(pdf_content)


def pdf_to_images(pdf_path='/tmp/raw_pdf/Budget_économique_prévisionnel_2024.pdf', output_dir='/tmp/budget_eco/2024/'):
    print('Converting PDF to images...')

    os.makedirs(output_dir, exist_ok=True)

    with open(pdf_path, 'rb') as pdf_file:
        pdf_content = pdf_file.read()

    try:
        images = convert_from_bytes(pdf_content, dpi=300)

        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'page_{i + 1}.jpg')
            image.save(image_path, 'JPEG')
            print(f"Saved image: {image_path}")

    except MemoryError:
        print(f"Erreur de mémoire: {MemoryError}")

def get_tables():
    kpi = ['annexe']
    search = 'agricole'
    image_dir = '/tmp/budget_eco/2024/'
    ocr_processor = ocr.OCRProcessor()
    tables = get_tables(ocr_processor, image_dir, kpi, search, reverse=True)
    logging(tables)

# def get_data_eco():
#     print('Obtention du contenu PDF ...')
#     pdf_content = asyncio.run(get_pdf_content())
#     if pdf_content:
#         print('Contenu PDF récupéré avec succès.')
#
#         print("_____________________________________TABLES__________________________________________________")
#         kpi = ['annexe']
#         search = 'agricole'
#         table = extract_table(pdf_content, kpi, search)
#         print(table)
#         print("_____________________________________TEXTE__________________________________________________")
#         keyword = 'compte courant'
#         text = extract_text(pdf_content, keyword)
#         print(text)
#     else:
#         print("Échec du téléchargement du contenu PDF.")

def load_connections():
    db.merge_conn(
        Connection(
            conn_id="t1",
            conn_type="kafka",
            extra=json.dumps({"socket.timeout.ms": 10, "bootstrap.servers": "broker:29092"}),
        )
    )


def producer_function():
    data = get_test_data(file_path='data.csv')
    for i, item in enumerate(data):
        yield (json.dumps(i), json.dumps(item))

def put_file_to_hdfs():
    client = InsecureClient('http://namenode:9870', user='hdfs')
    local_path = '/tmp/test_data.csv'
    hdfs_path = '/hadoop/dfs/name/test_data.csv'
    client.upload(hdfs_path, local_path, overwrite=True)


def read_csv_to_dataframe():
    client = InsecureClient('http://namenode:9870', user='hdfs')
    hdfs_path = '/hadoop/dfs/name/test_data.csv'

    # Read the CSV file content into a StringIO object
    with client.read(hdfs_path, encoding='utf-8') as reader:
        csv_data = reader.read()

    # Use StringIO to turn the string data into a file-like object for pandas
    data = StringIO(csv_data)
    df = pd.read_csv(data)
    records = df.to_dict(orient='records')
    for item in records:
        yield (json.dumps(str(uuid.uuid4())), json.dumps(item))


with (DAG('user_automation_test_v1',
         default_args=default_args,
         schedule='@daily',
         catchup=False) as dag):

    # t00 = PythonOperator(task_id='fetch_get_data_eco',
    #                     python_callable=get_data_eco,
    #                     execution_timeout=timedelta(hours=3),
    #                     retries=2,
    #                     )

    t0 = PythonOperator(task_id='get_pdf',
                        python_callable=get_pdf,
                        execution_timeout=timedelta(hours=3),
                        )

    t1 = PythonOperator(task_id='pdf_to_images',
                        python_callable=pdf_to_images,
                        execution_timeout=timedelta(hours=3),
                        )

    t2 = PythonOperator(task_id='get_tables',
                        python_callable=get_tables,
                        execution_timeout=timedelta(hours=3),
                        )

    # t3 = PythonOperator(
    #     task_id='transfer_to_hdfs',
    #     python_callable=put_file_to_hdfs
    # )
    #
    # t3 = PythonOperator(task_id="load_connections", python_callable=load_connections)
    #
    # t4 = ProduceToTopicOperator(
    #     kafka_config_id="t1",
    #     task_id="produce_to_topic",
    #     topic="topic-1",
    #     producer_function=read_csv_to_dataframe,
    # )
    t0 >> t1 >> t2