import uuid
from argparse import ArgumentParser, RawTextHelpFormatter
import uuid
from asyncpg.exceptions import SerializationError

from db.dbSession import databaseInstance as db


async def register_user():

    if not db._connection_pool:
        await db.connect()

    con = await db._connection_pool.acquire()

    try:
        id = uuid.uuid4()
        await con.execute(
            "INSERT INTO users (id, username, password) VALUES ($1, $2, $3)",
            id,
            "abhinav",
            "abhinav@home2",
        )
        print("register_user(): User registered successfully")

    except Exception as e:
        print(f"register_user(): {str(e)}")
    finally:
        await db._connection_pool.release(con)
