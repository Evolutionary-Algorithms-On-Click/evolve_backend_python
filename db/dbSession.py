from db.dbPool import Database

from dotenv import load_dotenv

load_dotenv()

import os

connectionString = os.getenv(
    "DB_URL", "postgresql://postgres:password@localhost:5432/evolve"
) + os.getenv("DB_URL_PARAMS", "?sslmode=disable")

databaseInstance = Database(connectionString)


async def initDatabase():
    if not databaseInstance._connection_pool:
        await databaseInstance.connect()

    con = await databaseInstance._connection_pool.acquire()

    try:
        await con.execute(
            """
            DROP TABLE IF EXISTS users
            """
        )
        await con.execute(
            """
            DROP TABLE IF EXISTS otp
            """
        )
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                userId UUID PRIMARY KEY,
                userName VARCHAR(255) NOT NULL,
                userEmail VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                accountStatus VARCHAR(255) NOT NULL,
                CHECK (accountStatus IN ('REGISTERED', 'ACTIVE', 'BLOCKED'))
            )
            """
        )
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS otp(
                userId UUID PRIMARY KEY,
                otp CHAR(6) NOT NULL
            )
            """
        )

        print("initDatabase(): Tables created successfully")

    except Exception as e:
        print(f"{str(e)}")
    finally:
        await databaseInstance._connection_pool.release(con)
