# db.py
import os
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

def get_connection():
    """
    Connexion au serveur PostgreSQL .
    """
    return psycopg2.connect(
        host="avo-adb-002.postgres.database.azure.com",
        port=5432,
        database="Action Plan",
        user="administrationSTS",
        password="St$@0987",
        sslmode="require"
    )