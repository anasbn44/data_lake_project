import logging

from cassandra.cluster import Cluster
from cassandra.cqlengine import connection
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

def create_topic_if_not_exist(bootstrap_servers, topic_name, num_partitions = 1, replication_factor = 1):
    # Initialize Kafka Admin Client
    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id="kafka-admin"
    )

    # Create a NewTopic object with desired properties
    new_topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )

    # Check if the topic exists or create it
    try:
        existing_topics = admin_client.list_topics()
        if topic_name not in existing_topics:
            # Create the topic if it doesn't exist
            admin_client.create_topics([new_topic])
            print(f"Topic '{topic_name}' created successfully.")
        else:
            print(f"Topic '{topic_name}' already exists.")
    except TopicAlreadyExistsError:
        print(f"Topic '{topic_name}' already exists.")
    except Exception as e:
        print(f"Error creating Kafka topic: {e}")
    finally:
        admin_client.close()

def create_cassandra_keyspace(session, keyspace, replication_factor=1):
    """
    Creates a Cassandra keyspace if it doesn't exist.

    :param session: Cassandra session object.
    :param keyspace: The name of the keyspace to create.
    :param replication_factor: The replication factor for the keyspace.
    """
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': {replication_factor} }}
    """)
    print(f"Keyspace '{keyspace}' ensured/created successfully.")

def create_cassandra_table(session, keyspace, table):
    """
    Creates a Cassandra table if it doesn't exist.

    :param session: Cassandra session object.
    :param keyspace: The keyspace in which the table will be created.
    :param table: The name of the table to create.
    """
    session.execute(f"""
        CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
            key text PRIMARY KEY,
            value text
        )
    """)
    print(f"Table '{table}' ensured/created successfully in keyspace '{keyspace}'.")

def setup_cassandra(cassandra_host : list, keyspace, table):
    """
    Initialize Cassandra Cluster, create keyspace and table.

    :param cassandra_host: List of Cassandra hosts.
    :param keyspace: Keyspace to create if not existing.
    :param table: Table to create if not existing.
    """
    # Initialize Cassandra Cluster and Session
    cluster = Cluster(cassandra_host)
    session = cluster.connect()

    # Create the keyspace and table
    create_cassandra_keyspace(session, keyspace)
    create_cassandra_table(session, keyspace, table)

    # Optionally return the session object if needed
    return session


def insert_data(session, **kwargs):
    print("inserting data...")

    id = kwargs.get('id')
    order_date = kwargs.get('order_date')
    product_name = kwargs.get('product_name')
    quantity = kwargs.get('quantity')

    try:
        session.execute("""
            INSERT INTO spark_streams.created_users(id, order_date, product_name, quantity)
                VALUES (%s, %s, %s, %s)
        """, (id, order_date, product_name, quantity))
        logging.info(f"Data inserted for {product_name} {quantity}")

    except Exception as e:
        logging.error(f'could not insert data due to {e}')


def create_spark_connection():
    try:
        # Spark session is established with cassandra and kafka jars. Suitable versions can be found in Maven repository.
        spark = SparkSession \
                .builder \
                .appName("SparkStructuredStreaming") \
                .config("spark.master", "spark://spark-master:7077")\
                .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.4.0,spark-sql-kafka-0-10_2.12-3.4.3") \
                .config("spark.cassandra.connection.host", "cassandra") \
                .config("spark.cassandra.connection.port","9042")\
                .config("spark.cassandra.auth.username", "cassandra") \
                .config("spark.cassandra.auth.password", "cassandra") \
                .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        logging.info('Spark session created successfully')
    except Exception:
        logging.error("Couldn't create the spark session")

    return spark


def connect_to_kafka(spark_conn):
    spark_df = None
    try:
        spark_df = spark_conn.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'broker:29092') \
            .option('subscribe', 'topic-1') \
            .load()
        logging.info("kafka dataframe created successfully")
    except Exception as e:
        logging.warning(f"kafka dataframe could not be created because: {e}")

    return spark_df


def create_cassandra_connection():
    try:
        # connecting to the cassandra cluster
        cluster = Cluster(['cassandra'])
        cass_session = cluster.connect()

        return cass_session
    except Exception as e:
        logging.error(f"Could not create cassandra connection due to {e}")
        return None


def create_selection_df_from_kafka(spark_df):
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("order_date", StringType(), False),
        StructField("product_name", StringType(), False),
        StructField("quantity", StringType(), False),
    ])

    sel = spark_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col('value'), schema).alias('data')).select("data.*")
    print(sel)

    return sel

def save_to_cassandra(batch_df, batch_id, table, keyspace):
    batch_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=table, keyspace=keyspace) \
        .mode("append") \
        .save()


if __name__ == "__main__":

    create_topic_if_not_exist("broker:29092", "topic-1")

    spark = SparkSession \
        .builder \
        .appName("SparkStructuredStreaming") \
        .config("spark.master", "spark://spark-master:7077") \
        .config("spark.jars.packages",
                "com.datastax.spark:spark-cassandra-connector_2.12:3.4.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.3") \
        .config("spark.cassandra.connection.host", "cassandra") \
        .config("spark.cassandra.connection.port", "9042") \
        .config("spark.cassandra.auth.username", "cassandra") \
        .config("spark.cassandra.auth.password", "cassandra") \
        .getOrCreate()

    print("************************************\n"+
          "************************************\n"+
          "************************************\n"+
          "************************************\n")

    schema = StructType() \
        .add("key", StringType()) \
        .add("value", StringType())

    kafka_df = spark \
          .readStream \
          .format("kafka") \
          .option("kafka.bootstrap.servers", "broker:29092") \
          .option("subscribe", "topic-1") \
          .load()

    messages_df = kafka_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

    session = setup_cassandra(['cassandra'], 'test_keyspace', 'test_table')

    def write_to_cassandra(batch_df, batch_id):
        save_to_cassandra(batch_df, batch_id, "test_table", "test_keyspace")
    # Start the streaming query and print incoming messages to the console
    query = messages_df \
        .writeStream \
        .foreachBatch(write_to_cassandra) \
        .outputMode("append") \
        .start()

    query.awaitTermination()