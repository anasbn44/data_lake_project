### Init airflow
```
mkdir ./dags, ./logs, ./plugins, ./config
```

```
docker compose up airflow-init
```

#### To stop and delete containers, delete volumes with database data and download images, run:
```
docker compose down --volumes --rmi all
docker stats --no-stream
```