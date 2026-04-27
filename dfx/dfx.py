"""
DFx: a lightweight, chain-friendly DataFrame utility wrapper.

This module provides :class:`DFx`, a thin wrapper around PySpark
``DataFrame`` objects. It is intended for ETL code that repeatedly needs small
schema and column hygiene helpers but should not monkey-patch Spark's
``DataFrame`` class or hide normal Spark behavior.

Design goals
------------
DFx follows Spark's immutable transformation style. Methods that transform the
wrapped DataFrame return a new ``DFx`` instance, and the underlying DataFrame is
available at any time through :meth:`DFx.unwrap`.

The module keeps two styles of helpers:

* Chain-friendly instance methods, such as ``normalize_column`` and ``flatten``.
* Static utility methods, such as ``clean_columns`` and ``merge_dataframes``,
  for cases where wrapping a DataFrame is unnecessary or inconvenient.

Basic usage
-----------
Wrap a Spark DataFrame, chain transformations, then unwrap it when you need a
plain PySpark DataFrame:

.. code-block:: python

    from dfx import DFx
    from pyspark.sql import functions as F

    result_df = (
        DFx(raw_df)
        .clean_column_names()
        .normalize_column("status", "UNKNOWN", "string")
        .cast_if_exists("amount", "decimal(18,2)")
        .withColumn("processed_at", F.current_timestamp())
        .unwrap()
    )

Joining DataFrames
------------------
Use the static ``merge_dataframes`` helper when you want a plain DataFrame
back, or the instance ``merge`` helper when building a DFx chain:

.. code-block:: python

    joined_df = DFx.merge_dataframes(
        orders_df,
        customers_df,
        left_keys="customer_id",
        right_keys="id",
        how="left",
        drop_side="right",
    )

    chained_df = (
        DFx(orders_df)
        .merge(customers_df, left_keys="customer_id", right_keys="id")
        .normalize_column("customer_status", "UNKNOWN", "string")
        .unwrap()
    )

Schema alignment
----------------
Align an incoming DataFrame to an existing Hive or Delta table before append:

.. code-block:: python

    aligned_df = (
        DFx(incoming_df)
        .align_to_table_schema("analytics.fact_orders", spark)
        .unwrap()
    )

Notes
-----
DFx deliberately exposes only a small proxy surface for common DataFrame
methods. For anything not proxied, call :meth:`DFx.unwrap` and continue with
regular PySpark APIs.
"""

from __future__ import annotations

import re
from functools import reduce
from typing import Any, Iterable

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


class DFx:
    """
    Thin, immutable-style wrapper for PySpark DataFrames.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame to wrap. DFx stores this object as ``self.df`` and never
        mutates it in place. Every transformation method creates and returns a
        new ``DFx`` instance around Spark's returned DataFrame.

    Examples
    --------
    Build a compact ETL chain:

    .. code-block:: python

        cleaned = (
            DFx(raw_df)
            .clean_column_names()
            .normalize_column("load_status", "NEW", "string")
            .flatten(separator="_")
            .unwrap()
        )

    Mix DFx helpers with normal Spark expressions:

    .. code-block:: python

        from pyspark.sql import functions as F

        result = (
            DFx(raw_df)
            .withColumn("event_date", DFx.safe_parse_date("event_date_raw"))
            .filter(F.col("event_date").isNotNull())
            .select("id", "event_date")
            .unwrap()
        )

    Use static helpers without wrapping:

    .. code-block:: python

        renamed_columns = DFx.clean_columns(source_df.columns)
        source_df = source_df.toDF(*renamed_columns)
    """

    def __init__(self, df: DataFrame):
        """
        Initialize a DFx wrapper.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame to wrap.
        """
        self.df = df

    @staticmethod
    def _normalize_keys(keys: str | Iterable[str]) -> list[str]:
        """
        Convert a single key or iterable of keys into a list.

        Parameters
        ----------
        keys : str or Iterable[str]
            A single column name or an ordered collection of column names.

        Returns
        -------
        list[str]
            Ordered list of key names.

        Notes
        -----
        This is an internal helper used by join methods. It treats strings as a
        single key instead of iterating over individual characters.
        """
        if isinstance(keys, str):
            return [keys]
        return list(keys)

    @staticmethod
    def _quote_name(name: str) -> str:
        """
        Escape a Spark column name for use in ``F.col`` expressions.

        Parameters
        ----------
        name : str
            A literal top-level column name.

        Returns
        -------
        str
            Backtick-escaped column reference.

        Notes
        -----
        Spark interprets dots in unescaped column names as nested field access.
        This helper quotes the entire top-level name so columns such as
        ``"order.id"`` are treated as literal column names.
        """
        return f"`{name.replace('`', '``')}`"

    @staticmethod
    def _missing_columns(df: DataFrame, columns: Iterable[str]) -> list[str]:
        """
        Return requested columns that are absent from a DataFrame.

        This intentionally checks top-level column names only, matching the
        rest of DFx's schema hygiene helpers.
        """
        available_columns = set(df.columns)
        return [column for column in columns if column not in available_columns]

    @staticmethod
    def _is_left_only_join(how: str) -> bool:
        """
        Return whether a Spark join type projects only left-side columns.
        """
        normalized_how = how.lower().replace(" ", "").replace("-", "_")
        return normalized_how in {
            "semi",
            "anti",
            "left_semi",
            "leftsemi",
            "left_anti",
            "leftanti",
        }

    def normalize_column(
        self,
        column_name: str,
        default_value: Any,
        cast_type: str | T.DataType,
    ) -> "DFx":
        """
        Ensure a column exists, contains no nulls, and is cast to a target type.

        If the column is missing, it is created from ``default_value``. If the
        column exists, null values are replaced with ``default_value``. In both
        cases the resulting column is cast to ``cast_type``.

        Parameters
        ----------
        column_name : str
            Name of the column to normalize.
        default_value : Any
            Value used for missing columns or null values.
        cast_type : str or pyspark.sql.types.DataType
            Spark SQL type used for the final cast, such as ``"string"``,
            ``"int"``, ``"decimal(18,2)"``, or ``T.IntegerType()``.

        Returns
        -------
        DFx
            New wrapper around the transformed DataFrame.

        Examples
        --------
        .. code-block:: python

            df = (
                DFx(raw_df)
                .normalize_column("status", "UNKNOWN", "string")
                .normalize_column("amount", 0, "decimal(18,2)")
                .unwrap()
            )
        """
        df = self.df

        if column_name not in df.columns:
            df = df.withColumn(column_name, F.lit(default_value).cast(cast_type))
        else:
            quoted_name = self._quote_name(column_name)
            df = df.withColumn(
                column_name,
                F.when(F.col(quoted_name).isNull(), F.lit(default_value))
                .otherwise(F.col(quoted_name))
                .cast(cast_type),
            )

        return DFx(df)

    @staticmethod
    def merge_dataframes(
        left_df: DataFrame,
        right_df: DataFrame,
        left_keys: str | Iterable[str],
        right_keys: str | Iterable[str] | None = None,
        how: str = "left",
        drop_side: str = "right",
    ) -> DataFrame:
        """
        Join two DataFrames while controlling duplicate non-key columns.

        This helper supports both same-name and different-name join keys. It
        also provides deterministic handling for overlapping non-key columns by
        dropping the overlap from the selected side before the join.

        Parameters
        ----------
        left_df : pyspark.sql.DataFrame
            Left side of the join.
        right_df : pyspark.sql.DataFrame
            Right side of the join.
        left_keys : str or Iterable[str]
            Left-side join key or ordered join keys.
        right_keys : str or Iterable[str], optional
            Right-side join key or ordered join keys. Defaults to
            ``left_keys`` when omitted.
        how : str, default "left"
            Spark join type, such as ``"left"``, ``"inner"``, ``"right"``,
            ``"outer"``, ``"left_semi"``, or ``"left_anti"``.
        drop_side : {"right", "left", "none"}, default "right"
            Which side should lose overlapping non-key columns before the join.
            ``"none"`` leaves overlaps in place, which can produce duplicate
            output column names.

        Returns
        -------
        pyspark.sql.DataFrame
            Joined Spark DataFrame.

        Raises
        ------
        ValueError
            If ``left_keys`` and ``right_keys`` have different lengths, or if
            ``drop_side`` is not one of the supported values.

        Examples
        --------
        Join with matching key names:

        .. code-block:: python

            joined = DFx.merge_dataframes(
                left_df=orders_df,
                right_df=customers_df,
                left_keys="customer_id",
            )

        Join with different key names:

        .. code-block:: python

            joined = DFx.merge_dataframes(
                orders_df,
                customers_df,
                left_keys="customer_id",
                right_keys="id",
                how="left",
                drop_side="right",
            )

        Notes
        -----
        When key names differ, right-side key columns are omitted from the
        result so the output keeps the left-side key naming convention.
        """
        left_keys = DFx._normalize_keys(left_keys)
        right_keys = left_keys if right_keys is None else DFx._normalize_keys(right_keys)

        if len(left_keys) != len(right_keys):
            raise ValueError("left_keys and right_keys must have the same length")

        if not left_keys:
            raise ValueError("At least one join key is required")

        if drop_side not in {"left", "right", "none"}:
            raise ValueError("drop_side must be one of: 'left', 'right', 'none'")

        missing_left_keys = DFx._missing_columns(left_df, left_keys)
        missing_right_keys = DFx._missing_columns(right_df, right_keys)
        if missing_left_keys or missing_right_keys:
            details = []
            if missing_left_keys:
                details.append(f"left keys missing: {missing_left_keys}")
            if missing_right_keys:
                details.append(f"right keys missing: {missing_right_keys}")
            raise ValueError("; ".join(details))

        left_key_set = set(left_keys)
        right_key_set = set(right_keys)

        overlapping_non_keys = [
            col
            for col in set(left_df.columns).intersection(right_df.columns)
            if col not in left_key_set and col not in right_key_set
        ]

        if drop_side == "right":
            right_df = right_df.drop(*overlapping_non_keys)
        elif drop_side == "left":
            left_df = left_df.drop(*overlapping_non_keys)

        if left_keys == right_keys:
            return left_df.join(right_df, on=left_keys, how=how)

        left_alias = "__dfx_left"
        right_alias = "__dfx_right"
        left = left_df.alias(left_alias)
        right = right_df.alias(right_alias)

        join_condition = reduce(
            lambda cond, pair: cond
            & (
                F.col(f"{left_alias}.{DFx._quote_name(pair[0])}")
                == F.col(f"{right_alias}.{DFx._quote_name(pair[1])}")
            ),
            zip(left_keys, right_keys),
            F.lit(True),
        )

        joined_df = left.join(right, on=join_condition, how=how)

        selected_columns = [
            F.col(f"{left_alias}.{DFx._quote_name(col)}").alias(col)
            for col in left_df.columns
        ]
        if DFx._is_left_only_join(how):
            return joined_df.select(*selected_columns)

        selected_columns.extend(
            F.col(f"{right_alias}.{DFx._quote_name(col)}").alias(col)
            for col in right_df.columns
            if col not in right_key_set
        )

        return joined_df.select(*selected_columns)

    def merge(
        self,
        right_df: DataFrame | "DFx",
        left_keys: str | Iterable[str],
        right_keys: str | Iterable[str] | None = None,
        how: str = "left",
        drop_side: str = "right",
    ) -> "DFx":
        """
        Chain-friendly wrapper around :meth:`merge_dataframes`.

        Parameters
        ----------
        right_df : pyspark.sql.DataFrame or DFx
            Right side of the join. If a ``DFx`` instance is supplied, it is
            unwrapped automatically.
        left_keys : str or Iterable[str]
            Left-side join key or ordered join keys from the wrapped DataFrame.
        right_keys : str or Iterable[str], optional
            Right-side join key or ordered join keys. Defaults to
            ``left_keys``.
        how : str, default "left"
            Spark join type.
        drop_side : {"right", "left", "none"}, default "right"
            Which side should lose overlapping non-key columns before the join.

        Returns
        -------
        DFx
            New wrapper around the joined DataFrame.

        Examples
        --------
        .. code-block:: python

            result = (
                DFx(orders_df)
                .merge(customers_df, left_keys="customer_id", right_keys="id")
                .normalize_column("customer_status", "UNKNOWN", "string")
                .unwrap()
            )
        """
        right = right_df.unwrap() if isinstance(right_df, DFx) else right_df
        return DFx(
            self.merge_dataframes(
                self.df,
                right,
                left_keys=left_keys,
                right_keys=right_keys,
                how=how,
                drop_side=drop_side,
            )
        )

    @staticmethod
    def clean_columns(columns: Iterable[str]) -> list[str]:
        """
        Normalize column names by removing unsupported characters and resolving duplicates.

        The current normalization preserves the original utility behavior:
        leading and trailing whitespace are stripped, the characters space,
        dot, parentheses, and slash are removed, empty names become ``"col"``,
        and duplicate cleaned names receive ``_2``, ``_3``, and so on.

        Parameters
        ----------
        columns : Iterable[str]
            Original column names.

        Returns
        -------
        list[str]
            Cleaned column names in the same order as the input.

        Examples
        --------
        .. code-block:: python

            DFx.clean_columns([" Account ID ", "Account.ID", ""])
            # ["AccountID", "AccountID_2", "col"]

            cleaned_df = raw_df.toDF(*DFx.clean_columns(raw_df.columns))
        """
        cleaned_columns = []
        seen_names: dict[str, int] = {}

        for column_name in columns:
            normalized_name = str(column_name).strip()
            normalized_name = re.sub(r"[ .()/]", "", normalized_name)

            if normalized_name == "":
                normalized_name = "col"

            if normalized_name in seen_names:
                seen_names[normalized_name] += 1
                cleaned_name = f"{normalized_name}_{seen_names[normalized_name]}"
            else:
                seen_names[normalized_name] = 1
                cleaned_name = normalized_name

            cleaned_columns.append(cleaned_name)

        return cleaned_columns

    def clean_column_names(self) -> "DFx":
        """
        Rename the wrapped DataFrame columns using :meth:`clean_columns`.

        Returns
        -------
        DFx
            New wrapper around a DataFrame with cleaned top-level column names.

        Examples
        --------
        .. code-block:: python

            cleaned_df = DFx(raw_df).clean_column_names().unwrap()

        Notes
        -----
        This method uses ``DataFrame.toDF`` and only changes top-level column
        names. It does not rename nested struct fields.
        """
        return DFx(self.df.toDF(*self.clean_columns(self.df.columns)))

    @staticmethod
    def change_schema_field_type(
        schema: T.StructType,
        field_name: str,
        new_type: T.DataType,
    ) -> T.StructType:
        """
        Return a new schema with a specific top-level field updated.

        Parameters
        ----------
        schema : pyspark.sql.types.StructType
            Source schema.
        field_name : str
            Top-level field name whose type should be replaced.
        new_type : pyspark.sql.types.DataType
            Replacement Spark SQL data type.

        Returns
        -------
        pyspark.sql.types.StructType
            New schema with the requested field type changed. Field nullability
            and metadata are preserved.

        Examples
        --------
        .. code-block:: python

            new_schema = DFx.change_schema_field_type(
                raw_df.schema,
                "amount",
                T.DecimalType(18, 2),
            )

        Notes
        -----
        If ``field_name`` is not present, the returned schema is structurally
        equivalent to the input schema.
        """
        return T.StructType(
            [
                T.StructField(field.name, new_type, field.nullable, field.metadata)
                if field.name == field_name
                else field
                for field in schema.fields
            ]
        )

    def align_to_table_schema(
        self,
        table_name: str,
        spark: SparkSession,
    ) -> "DFx":
        """
        Align the wrapped DataFrame to match the schema of a Hive/Delta table.

        The target schema is read from ``spark.table(table_name).schema``.
        Missing target columns are added as typed nulls, existing target
        columns are cast to the table data types, extra input columns are
        dropped, and the output column order matches the target table.

        Parameters
        ----------
        table_name : str
            Fully qualified or session-resolvable table name.
        spark : pyspark.sql.SparkSession
            Active Spark session used to read the target table schema.

        Returns
        -------
        DFx
            New wrapper around a DataFrame aligned to the target table schema.

        Examples
        --------
        .. code-block:: python

            aligned_df = (
                DFx(incoming_df)
                .align_to_table_schema("analytics.fact_orders", spark)
                .unwrap()
            )

            aligned_df.write.mode("append").saveAsTable("analytics.fact_orders")

        Notes
        -----
        This method is intended for flat table schemas. It does not recursively
        align nested struct fields, and it relies on Spark's normal cast
        behavior for incompatible values.
        """
        target_schema = spark.table(table_name).schema
        source_columns = set(self.df.columns)

        aligned_columns = [
            (
                F.col(self._quote_name(field.name)).cast(field.dataType)
                if field.name in source_columns
                else F.lit(None).cast(field.dataType)
            ).alias(field.name)
            for field in target_schema
        ]
        return DFx(
            self.df.select(*aligned_columns)
        )

    @staticmethod
    def safe_parse_date(col: Column | str, formats: Iterable[str] | None = None) -> Column:
        """
        Attempt to parse a column using multiple date formats.

        The return value preserves the original helper's fallback behavior: if
        all parse attempts fail, the original column value is returned.

        Parameters
        ----------
        col : pyspark.sql.Column or str
            Source column expression or column name containing date-like values.
        formats : Iterable[str], optional
            Spark datetime format strings to try in order. Defaults to common
            ISO, slash-delimited, compact, and month-first patterns.

        Returns
        -------
        pyspark.sql.Column
            Column expression containing the first successfully parsed date,
            cast to string, or the original value cast to string when parsing
            fails.

        Examples
        --------
        .. code-block:: python

            parsed_df = raw_df.withColumn(
                "event_date",
                DFx.safe_parse_date("event_date_raw"),
            )

            parsed_df = raw_df.withColumn(
                "event_date",
                DFx.safe_parse_date(
                    F.col("event_date_raw"),
                    formats=["yyyy-MM-dd", "MM/dd/yyyy"],
                ),
            )

        Notes
        -----
        The expression uses ``try_to_timestamp`` when available and falls back
        to ``to_date`` otherwise. It returns a string to preserve the prior
        fallback behavior of returning the original source value on parse
        failure. If a strict ``DateType`` output is required, remove the final
        fallback and return the parsed expression directly.
        """
        if formats is None:
            formats = [
                "yyyy-MM-dd",
                "yyyy/MM/dd",
                "yyyyMMdd",
                "MM/dd/yyyy",
                "MM-dd-yyyy",
                "dd-MM-yyyy",
            ]

        source_col = F.col(col) if isinstance(col, str) else col
        parsed_expr = None
        use_try_to_timestamp = hasattr(F, "try_to_timestamp")

        for fmt in formats:
            if use_try_to_timestamp:
                attempt = F.try_to_timestamp(source_col, F.lit(fmt)).cast("date")
            else:
                attempt = F.to_date(source_col, fmt)
            parsed_expr = attempt if parsed_expr is None else F.coalesce(parsed_expr, attempt)

        if parsed_expr is None:
            return source_col.cast("string")

        return F.coalesce(parsed_expr.cast("string"), source_col.cast("string"))

    @staticmethod
    def safe_cast(df: DataFrame, col: str, dtype: str | T.DataType) -> DataFrame:
        """
        Safely cast a column to a new type only if the column exists.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame.
        col : str
            Top-level column name to cast.
        dtype : str or pyspark.sql.types.DataType
            Spark SQL target type.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the cast applied when ``col`` exists; otherwise the
            original DataFrame is returned unchanged.

        Examples
        --------
        .. code-block:: python

            df = DFx.safe_cast(raw_df, "amount", "decimal(18,2)")

        Notes
        -----
        This helper is useful for optional source columns in reusable ETL code.
        """
        if col in df.columns:
            return df.withColumn(col, F.col(DFx._quote_name(col)).cast(dtype))
        return df

    def cast_if_exists(self, col: str, dtype: str | T.DataType) -> "DFx":
        """
        Chain-friendly wrapper around :meth:`safe_cast`.

        Parameters
        ----------
        col : str
            Top-level column name to cast if present.
        dtype : str or pyspark.sql.types.DataType
            Spark SQL target type.

        Returns
        -------
        DFx
            New wrapper around the cast DataFrame, or around the unchanged
            DataFrame when the column does not exist.

        Examples
        --------
        .. code-block:: python

            df = (
                DFx(raw_df)
                .cast_if_exists("amount", "decimal(18,2)")
                .cast_if_exists("event_count", "int")
                .unwrap()
            )
        """
        return DFx(self.safe_cast(self.df, col, dtype))

    def flatten(self, separator: str = ".") -> "DFx":
        """
        Recursively flatten nested struct fields into top-level columns.

        Each non-struct field is projected into the result. Struct field paths
        become top-level aliases joined by ``separator``.

        Parameters
        ----------
        separator : str, default "."
            Separator used when constructing flattened output column names.

        Returns
        -------
        DFx
            New wrapper around the flattened DataFrame.

        Examples
        --------
        Given a schema like ``customer: struct<id:string, name:string>``:

        .. code-block:: python

            flattened_df = DFx(raw_df).flatten(separator="_").unwrap()
            # Output columns include: customer_id, customer_name

        Notes
        -----
        Arrays and maps are not exploded or expanded. They are preserved as
        leaf columns. Nested field references are backtick-escaped so unusual
        field names are handled more safely.
        """

        def _flatten_fields(
            schema: T.StructType,
            parent_parts: tuple[str, ...] = (),
            parent_alias: str = "",
        ) -> list[Column]:
            flattened_fields = []

            for field in schema.fields:
                source_parts = (*parent_parts, field.name)
                alias_name = f"{parent_alias}{separator}{field.name}" if parent_alias else field.name

                if isinstance(field.dataType, T.StructType):
                    flattened_fields.extend(
                        _flatten_fields(
                            schema=field.dataType,
                            parent_parts=source_parts,
                            parent_alias=alias_name,
                        )
                    )
                else:
                    source_path = ".".join(
                        f"`{part.replace('`', '``')}`" for part in source_parts
                    )
                    flattened_fields.append(F.col(source_path).alias(alias_name))

            return flattened_fields

        return DFx(self.df.select(*_flatten_fields(self.df.schema)))

    def withColumn(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.withColumn`` and wrap the returned DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.withColumn``.

        Returns
        -------
        DFx
            New wrapper around the transformed DataFrame.

        Examples
        --------
        .. code-block:: python

            df = (
                DFx(raw_df)
                .withColumn("loaded_at", F.current_timestamp())
                .unwrap()
            )
        """
        return DFx(self.df.withColumn(*args, **kwargs))

    def select(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.select`` and wrap the returned DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.select``.

        Returns
        -------
        DFx
            New wrapper around the projected DataFrame.

        Examples
        --------
        .. code-block:: python

            df = DFx(raw_df).select("id", "status").unwrap()
        """
        return DFx(self.df.select(*args, **kwargs))

    def filter(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.filter`` and wrap the returned DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.filter``.

        Returns
        -------
        DFx
            New wrapper around the filtered DataFrame.

        Examples
        --------
        .. code-block:: python

            df = DFx(raw_df).filter(F.col("status") == "ACTIVE").unwrap()
        """
        return DFx(self.df.filter(*args, **kwargs))

    def where(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.where`` and wrap the returned DataFrame.

        ``where`` is Spark's alias for ``filter`` and is included so DFx chains
        can use either spelling.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.where``.

        Returns
        -------
        DFx
            New wrapper around the filtered DataFrame.

        Examples
        --------
        .. code-block:: python

            df = DFx(raw_df).where("status = 'ACTIVE'").unwrap()
        """
        return DFx(self.df.where(*args, **kwargs))

    def drop(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.drop`` and wrap the returned DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.drop``.

        Returns
        -------
        DFx
            New wrapper around the DataFrame after dropping columns.

        Examples
        --------
        .. code-block:: python

            df = DFx(raw_df).drop("raw_payload", "debug_notes").unwrap()
        """
        return DFx(self.df.drop(*args, **kwargs))

    def withColumnRenamed(self, *args: Any, **kwargs: Any) -> "DFx":
        """
        Call ``DataFrame.withColumnRenamed`` and wrap the returned DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to ``pyspark.sql.DataFrame.withColumnRenamed``.

        Returns
        -------
        DFx
            New wrapper around the DataFrame with the renamed column.

        Examples
        --------
        .. code-block:: python

            df = DFx(raw_df).withColumnRenamed("acct_id", "account_id").unwrap()
        """
        return DFx(self.df.withColumnRenamed(*args, **kwargs))

    def unwrap(self) -> DataFrame:
        """
        Return the underlying Spark DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            The currently wrapped DataFrame.

        Examples
        --------
        .. code-block:: python

            spark_df = (
                DFx(raw_df)
                .normalize_column("status", "UNKNOWN", "string")
                .unwrap()
            )

        Notes
        -----
        Use ``unwrap`` when you need an unproxied Spark method, an action such
        as ``collect`` or ``write``, or an API that expects a plain DataFrame.
        """
        return self.df
