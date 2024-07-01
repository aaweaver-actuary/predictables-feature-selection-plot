"""Define a wrapper around the DuckDB database."""

from __future__ import annotations
import logging
from dataclasses import dataclass
import polars as pl
import duckdb

__all__ = ["DuckDB"]

DB_FILENAME = "data/db.duckdb"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    filename="duck_db.log",
)

logger = logging.getLogger(__name__)


@dataclass
class DuckDB:
    """Wrapper around the DuckDB database."""

    db_file: str = DB_FILENAME
    db_conn: duckdb.DuckDBPyConnection | None = None

    slots = "db_file"

    def __post_init__(self):
        """Initialize the DuckDB connection."""
        logger.debug(f"Creating {self.__class__.__name__} with db_file={self.db_file}")

    def __call__(self, query: str) -> pl.DataFrame:
        """Execute a query on the DuckDB database."""
        return self.read(query)

    def write(self, query: str) -> pl.DataFrame | None:
        """Execute a write operation on the DuckDB database."""
        logger.debug(f"query for write op:\n{query}")
        with duckdb.connect(self.db_file, read_only=False) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def read(self, query: str) -> pl.DataFrame | None:
        """Execute a read operation on the DuckDB database."""
        logger.debug(f"query for read op:\n{query}")
        with duckdb.connect(self.db_file, read_only=True) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def __enter__(self):
        """Enter a with context on the DuckDB database."""
        logger.debug(f"Entering a with context on {self.db_file}")
        self.db_conn = duckdb.connect(self.db_file)
        logger.debug("With context opened")
        return self.db_conn

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Exit a with context on the DuckDB database."""
        logger.debug(f"Closing a with context on {self.db_file}")
        self.db_conn.close()
        self.db_conn = None
        logger.debug("With context closed")

    def get_user_defined_enum_types(self) -> list[str]:
        """Get the list of user-defined enum types in the DuckDB database."""
        enum_qry = """
            select type_name
            from duckdb_types()
            where 
                (
                    ends_with(type_name, '__type')
                    or ends_with(type_name, 'type')
                )
                and logical_type='ENUM'
        """
        logger.debug(f"Query to get user-defined enum types:\n{enum_qry}")
        try:
            with duckdb.connect(self.db_file) as conn:
                ud_types__RAW = conn.sql(enum_qry)
                ud_types = (
                    ud_types__RAW.to_pandas()["type_name"].tolist()
                    if ud_types__RAW.shape[0] > 0
                    else []
                )

        except Exception as _:
            ud_types = []

        logger.debug(f"User-defined enum types:\n{ud_types}")
        return ud_types

    def clear_user_defined_enum_type(self, type_name: str) -> None:
        """Drop a user-defined enum type from the DuckDB database."""
        ud_types = self.get_user_defined_enum_types()
        if type_name not in ud_types:
            err_msg = (
                f"`{type_name}` does not appear in the list of user-defined enum types:"
            )
            err_msg += "\n"
            err_msg += "\n".join(ud_types)
            logger.error(err_msg)

        try:
            drop_qry = f"drop type {type_name}"
            logger.debug(f"Query to drop type {type_name}:\n{drop_qry}")
            self.write(drop_qry)
            logger.debug(f"Type {type_name} dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping type {type_name}:\n{e}")

    def clear_all_user_defined_enum_types(self) -> None:
        """Drop all user-defined enum types from the DuckDB database."""
        for t in self.get_user_defined_enum_types():
            self.clear_user_defined_enum_type(t)

    def create_or_replace_string_enum(
        self, old_col: str, new_col: str, table: str
    ) -> None:
        """Create or replace an enum type from a string column in a table."""
        type_name = f"{new_col}__type"

        logger.debug(
            f"Creating or replacing enum type {type_name} from {old_col} in {table}"
        )

        # Drop the enum type if it is already defined
        if type_name in self.get_user_defined_enum_types():
            logger.debug(
                f"Type {type_name} already exists. Dropping it before creating a new one"
            )
            self.clear_user_defined_enum_type(type_name)
        elif type_name.replace("__", "_") in self.get_user_defined_enum_types():
            logger.debug(
                f"Type {type_name} does not exist but {type_name.replace('__', '_')} does. Dropping it before creating a new one"
            )
            self.clear_user_defined_enum_type(type_name.replace("__", "_"))
        else:
            logger.debug(
                f"Type {type_name} does not already exist, so just need to create it."
            )

        # Create the new enum type from the values from the table
        create_type_qry = f"create type {type_name} as enum (select distinct {old_col} from {table} where {old_col} is not null);"  # noqa: S608
        logger.debug(f"Query to create type {type_name}:\n{create_type_qry}")
        try:
            self.write(create_type_qry)
            logger.debug(f"Type {type_name} created successfully")
        except Exception as e:
            logger.error(f"Error creating type {type_name}:\n{e}")
