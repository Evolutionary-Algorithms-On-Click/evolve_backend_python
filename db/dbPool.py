import asyncpg


class Database:
    def __init__(self, connectionString=None):
        self.connectionString = connectionString
        self._cursor = None
        self._connection_pool = None
        self.con = None

    async def connect(self):
        if not self._connection_pool:
            try:
                self._connection_pool = await asyncpg.create_pool(
                    dsn=self.connectionString,
                    min_size=1,
                    max_size=10,
                )

            except Exception as e:
                print(e)
