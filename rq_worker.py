"""Simple RQ worker starter for local use.

Run on the host where Redis is reachable (or in a worker container):
    python rq_worker.py

This file starts an RQ worker listening on the default queue and imports
the tasks module so job handlers are available.
"""
import os
import redis
from rq import Worker, Queue, Connection

listen = ['default']

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
