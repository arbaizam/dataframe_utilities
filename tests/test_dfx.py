from __future__ import annotations

from pyspark.sql import types as T

from dfx import DFx
import dfx.dfx as dfx_module


class FakeExpr:
    def __init__(self, value):
        self.value = value
        self.alias_name = None
        self.cast_type = None

    def alias(self, name):
        expr = FakeExpr(self.value)
        expr.alias_name = name
        return expr

    def cast(self, dtype):
        expr = FakeExpr(self.value)
        expr.cast_type = dtype
        return expr

    def isNull(self):
        return FakeExpr(("is_null", self.value))

    def otherwise(self, other):
        return FakeExpr(("otherwise", self.value, other))

    def __and__(self, other):
        return FakeExpr(("and", self.value, other))

    def __eq__(self, other):
        return FakeExpr(("eq", self.value, other))


class FakeFunctions:
    def col(self, value):
        return FakeExpr(("col", value))

    def lit(self, value):
        return FakeExpr(("lit", value))

    def when(self, condition, value):
        return FakeExpr(("when", condition, value))

    def coalesce(self, *values):
        return FakeExpr(("coalesce", values))

    def to_date(self, col, fmt):
        return FakeExpr(("to_date", col, fmt))

    def try_to_timestamp(self, col, fmt):
        return FakeExpr(("try_to_timestamp", col, fmt))


class FakeDataFrame:
    def __init__(self, columns, schema=None, calls=None):
        self.columns = list(columns)
        self.schema = schema or T.StructType(
            [T.StructField(col, T.StringType()) for col in self.columns]
        )
        self.calls = calls or []
        self.alias_name = None

    def _next(self, columns=None, schema=None, call=None):
        calls = [*self.calls]
        if call is not None:
            calls.append(call)
        return FakeDataFrame(
            self.columns if columns is None else columns,
            self.schema if schema is None else schema,
            calls,
        )

    def withColumn(self, name, expr):
        columns = [*self.columns]
        if name not in columns:
            columns.append(name)
        return self._next(columns=columns, call=("withColumn", name, expr))

    def toDF(self, *columns):
        return self._next(columns=columns, call=("toDF", columns))

    def select(self, *columns):
        selected = []
        for col in columns:
            if isinstance(col, list):
                selected.extend(col)
            else:
                selected.append(col)

        names = [
            getattr(col, "alias_name", None) or str(getattr(col, "value", col))
            for col in selected
        ]
        return self._next(columns=names, call=("select", tuple(selected)))

    def filter(self, *args, **kwargs):
        return self._next(call=("filter", args, kwargs))

    def where(self, *args, **kwargs):
        return self._next(call=("where", args, kwargs))

    def drop(self, *columns):
        return self._next(
            columns=[col for col in self.columns if col not in columns],
            call=("drop", columns),
        )

    def withColumnRenamed(self, old, new):
        return self._next(
            columns=[new if col == old else col for col in self.columns],
            call=("withColumnRenamed", old, new),
        )

    def alias(self, name):
        aliased = self._next(call=("alias", name))
        aliased.alias_name = name
        return aliased

    def join(self, other, on=None, how=None):
        return self._next(
            columns=[*self.columns, *other.columns],
            call=("join", other, on, how),
        )


class FakeSpark:
    def __init__(self, schema):
        self.schema = schema
        self.table_names = []

    def table(self, name):
        self.table_names.append(name)
        return FakeDataFrame([field.name for field in self.schema], schema=self.schema)


def _patch_spark_functions(monkeypatch):
    fake = FakeFunctions()
    monkeypatch.setattr(dfx_module.F, "col", fake.col)
    monkeypatch.setattr(dfx_module.F, "lit", fake.lit)
    monkeypatch.setattr(dfx_module.F, "when", fake.when)
    monkeypatch.setattr(dfx_module.F, "coalesce", fake.coalesce)
    monkeypatch.setattr(dfx_module.F, "to_date", fake.to_date)
    monkeypatch.setattr(dfx_module.F, "try_to_timestamp", fake.try_to_timestamp)


def test_init_and_unwrap_preserve_wrapped_dataframe():
    """
    What: Verifies DFx stores and returns the exact DataFrame object it wraps.
    Why: unwrap is the escape hatch back to normal PySpark APIs.
    Fails when: DFx copies, replaces, or hides the wrapped DataFrame.
    """
    df = FakeDataFrame(["id"])

    wrapped = DFx(df)

    assert wrapped.df is df
    assert wrapped.unwrap() is df


def test_static_key_normalization_accepts_string_and_iterable():
    """
    What: Verifies join keys are normalized from a string or iterable.
    Why: A single string must be treated as one key, not as characters.
    Fails when: String keys become character lists or iterable ordering changes.
    """
    assert DFx._normalize_keys("id") == ["id"]
    assert DFx._normalize_keys(("id", "date")) == ["id", "date"]


def test_quote_name_escapes_literal_column_names():
    """
    What: Verifies literal column names are wrapped and embedded backticks escaped.
    Why: Spark treats dots in unquoted names as nested-field navigation.
    Fails when: Dotted or backtick-containing column names are not safely quoted.
    """
    assert DFx._quote_name("order.id") == "`order.id`"
    assert DFx._quote_name("a`b") == "`a``b`"


def test_normalize_column_adds_missing_column(monkeypatch):
    """
    What: Verifies normalize_column creates a missing column with a typed default.
    Why: ETL pipelines often need optional source columns made explicit.
    Fails when: Missing columns are not added or the method stops returning DFx.
    """
    _patch_spark_functions(monkeypatch)
    df = FakeDataFrame(["id"])

    result = DFx(df).normalize_column("status", "UNKNOWN", "string")

    assert isinstance(result, DFx)
    assert result.unwrap().columns == ["id", "status"]
    assert result.unwrap().calls[-1][0:2] == ("withColumn", "status")


def test_normalize_column_replaces_existing_nulls(monkeypatch):
    """
    What: Verifies normalize_column rewrites an existing column in place.
    Why: Existing nullable columns should be standardized without changing schema order.
    Fails when: Existing columns are duplicated, dropped, or not cast.
    """
    _patch_spark_functions(monkeypatch)
    df = FakeDataFrame(["id", "status"])

    result = DFx(df).normalize_column("status", "UNKNOWN", "string")

    assert result.unwrap().columns == ["id", "status"]
    assert result.unwrap().calls[-1][0:2] == ("withColumn", "status")


def test_merge_dataframes_joins_matching_key_names():
    """
    What: Verifies merge_dataframes delegates to Spark's named-key join path.
    Why: Matching key names should preserve Spark's clean join behavior.
    Fails when: Same-name keys are converted to an explicit condition unnecessarily.
    """
    left = FakeDataFrame(["id", "left_value"])
    right = FakeDataFrame(["id", "right_value"])

    result = DFx.merge_dataframes(left, right, left_keys="id", how="inner")

    assert result.calls[-1][0] == "join"
    assert result.calls[-1][2] == ["id"]
    assert result.calls[-1][3] == "inner"


def test_merge_dataframes_drops_overlapping_right_columns():
    """
    What: Verifies overlapping non-key columns are dropped from the right side by default.
    Why: The default join output should avoid duplicate non-key columns.
    Fails when: Right-side duplicate columns remain in the joined schema.
    """
    left = FakeDataFrame(["id", "value"])
    right = FakeDataFrame(["id", "value", "description"])

    result = DFx.merge_dataframes(left, right, left_keys="id")

    joined_right = result.calls[-1][1]
    assert joined_right.columns == ["id", "description"]


def test_merge_dataframes_supports_different_key_names(monkeypatch):
    """
    What: Verifies merge_dataframes supports joins where key names differ.
    Why: Source systems frequently use different names for the same join key.
    Fails when: The result includes the right-side key or omits right non-key columns.
    """
    _patch_spark_functions(monkeypatch)
    left = FakeDataFrame(["customer_id", "amount"])
    right = FakeDataFrame(["id", "segment"])

    result = DFx.merge_dataframes(left, right, left_keys="customer_id", right_keys="id")

    assert result.columns == ["customer_id", "amount", "segment"]
    assert result.calls[-1][0] == "select"


def test_merge_dataframes_validates_key_lengths_and_drop_side():
    """
    What: Verifies merge_dataframes rejects invalid join arguments.
    Why: Mismatched key lengths and unknown drop policies produce ambiguous joins.
    Fails when: Invalid join configuration is allowed through silently.
    """
    left = FakeDataFrame(["id"])
    right = FakeDataFrame(["id"])

    try:
        DFx.merge_dataframes(left, right, left_keys=["id", "date"], right_keys=["id"])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched key lengths")

    try:
        DFx.merge_dataframes(left, right, left_keys="id", drop_side="middle")
    except ValueError as exc:
        assert "drop_side" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid drop_side")


def test_merge_dataframes_validates_empty_and_missing_keys():
    """
    What: Verifies merge_dataframes rejects joins that cannot be resolved.
    Why: Missing key errors should point at DFx configuration, not Spark analysis.
    Fails when: Empty or absent join keys reach Spark and fail later.
    """
    left = FakeDataFrame(["id"])
    right = FakeDataFrame(["other_id"])

    try:
        DFx.merge_dataframes(left, right, left_keys=[])
    except ValueError as exc:
        assert "At least one join key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty keys")

    try:
        DFx.merge_dataframes(left, right, left_keys="missing", right_keys="other_id")
    except ValueError as exc:
        assert "left keys missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing left key")

    try:
        DFx.merge_dataframes(left, right, left_keys="id", right_keys="missing")
    except ValueError as exc:
        assert "right keys missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing right key")


def test_merge_dataframes_left_only_join_with_different_keys_projects_left(monkeypatch):
    """
    What: Verifies semi/anti joins with different key names project only left columns.
    Why: Spark semi/anti joins do not expose right-side columns for selection.
    Fails when: DFx attempts to select right columns after a left-only join.
    """
    _patch_spark_functions(monkeypatch)
    left = FakeDataFrame(["customer_id", "amount"])
    right = FakeDataFrame(["id", "segment"])

    result = DFx.merge_dataframes(
        left,
        right,
        left_keys="customer_id",
        right_keys="id",
        how="left_semi",
    )

    assert result.columns == ["customer_id", "amount"]
    assert result.calls[-1][0] == "select"


def test_merge_wraps_static_join_result(monkeypatch):
    """
    What: Verifies the instance merge method accepts DFx inputs and returns DFx.
    Why: Fluent pipelines should be able to join without unwrapping manually.
    Fails when: Instance merge returns a plain DataFrame or cannot unwrap right_df.
    """
    _patch_spark_functions(monkeypatch)
    left = DFx(FakeDataFrame(["customer_id"]))
    right = DFx(FakeDataFrame(["id", "segment"]))

    result = left.merge(right, left_keys="customer_id", right_keys="id")

    assert isinstance(result, DFx)
    assert result.unwrap().columns == ["customer_id", "segment"]


def test_clean_columns_preserves_existing_behavior():
    """
    What: Verifies clean_columns strips supported characters and resolves duplicates.
    Why: Existing column-cleaning behavior should remain stable for callers.
    Fails when: Cleaning rules or duplicate suffix numbering change.
    """
    columns = [" A.B ", "A/B", "", "A.B"]

    assert DFx.clean_columns(columns) == ["AB", "AB_2", "col", "AB_3"]


def test_clean_column_names_renames_wrapped_dataframe_columns():
    """
    What: Verifies clean_column_names applies clean_columns through DataFrame.toDF.
    Why: Chain-friendly column cleanup should mirror the static helper.
    Fails when: Cleaned names are not applied or the method returns a plain DataFrame.
    """
    df = FakeDataFrame([" A.B ", "A/B", ""])

    result = DFx(df).clean_column_names()

    assert isinstance(result, DFx)
    assert result.unwrap().columns == ["AB", "AB_2", "col"]
    assert result.unwrap().calls[-1][0] == "toDF"


def test_change_schema_field_type_preserves_nullable_and_metadata():
    """
    What: Verifies a requested schema field type is replaced.
    Why: Schema manipulation must keep field metadata and nullability intact.
    Fails when: Metadata, nullability, or unrelated fields are altered.
    """
    schema = T.StructType(
        [
            T.StructField("id", T.StringType(), False, {"source": "raw"}),
            T.StructField("name", T.StringType(), True),
        ]
    )

    updated = DFx.change_schema_field_type(schema, "id", T.IntegerType())

    assert updated["id"].dataType == T.IntegerType()
    assert updated["id"].nullable is False
    assert updated["id"].metadata == {"source": "raw"}
    assert updated["name"].dataType == T.StringType()


def test_change_schema_field_type_leaves_missing_field_schema_equivalent():
    """
    What: Verifies changing a missing schema field leaves the schema equivalent.
    Why: The helper currently preserves no-op behavior for missing fields.
    Fails when: Missing fields raise unexpectedly or alter existing fields.
    """
    schema = T.StructType([T.StructField("id", T.StringType())])

    updated = DFx.change_schema_field_type(schema, "missing", T.IntegerType())

    assert updated == schema


def test_align_to_table_schema_adds_casts_drops_and_reorders(monkeypatch):
    """
    What: Verifies align_to_table_schema follows the target table schema.
    Why: Table loads need typed missing columns, casts, dropped extras, and stable order.
    Fails when: The table schema is not read or the final projection is not target ordered.
    """
    _patch_spark_functions(monkeypatch)
    target_schema = T.StructType(
        [
            T.StructField("id", T.IntegerType()),
            T.StructField("status", T.StringType()),
        ]
    )
    spark = FakeSpark(target_schema)
    df = FakeDataFrame(["status", "extra"])

    result = DFx(df).align_to_table_schema("db.target", spark)

    assert spark.table_names == ["db.target"]
    assert result.unwrap().columns == ["id", "status"]
    assert result.unwrap().calls[-1][0] == "select"


def test_safe_parse_date_uses_formats_and_fallback(monkeypatch):
    """
    What: Verifies safe_parse_date builds a multi-format parsing expression.
    Why: Date ingestion needs deterministic parsing with fallback to source values.
    Fails when: Formats are ignored or the returned expression is not coalesced.
    """
    _patch_spark_functions(monkeypatch)

    result = DFx.safe_parse_date("raw_date", formats=["yyyy-MM-dd", "MM/dd/yyyy"])

    assert isinstance(result, FakeExpr)
    assert result.value[0] == "coalesce"


def test_safe_parse_date_handles_empty_format_list(monkeypatch):
    """
    What: Verifies safe_parse_date returns the source column when no formats are supplied.
    Why: Empty configurable format lists should not crash an ETL job.
    Fails when: Empty formats produce None or raise during expression construction.
    """
    _patch_spark_functions(monkeypatch)

    result = DFx.safe_parse_date("raw_date", formats=[])

    assert isinstance(result, FakeExpr)
    assert result.cast_type == "string"


def test_safe_cast_casts_existing_column(monkeypatch):
    """
    What: Verifies safe_cast casts a present column.
    Why: Optional casts should apply when source data contains the column.
    Fails when: Present columns are left unchanged or the wrong column is rewritten.
    """
    _patch_spark_functions(monkeypatch)
    df = FakeDataFrame(["id", "amount"])

    result = DFx.safe_cast(df, "amount", "decimal(18,2)")

    assert result.calls[-1][0:2] == ("withColumn", "amount")


def test_safe_cast_returns_original_dataframe_when_column_missing(monkeypatch):
    """
    What: Verifies safe_cast is a no-op for missing columns.
    Why: Reusable ETL code should tolerate optional columns.
    Fails when: Missing optional columns are added, dropped, or raise errors.
    """
    _patch_spark_functions(monkeypatch)
    df = FakeDataFrame(["id"])

    result = DFx.safe_cast(df, "amount", "decimal(18,2)")

    assert result is df


def test_cast_if_exists_wraps_safe_cast_result(monkeypatch):
    """
    What: Verifies cast_if_exists returns a DFx wrapper around safe_cast output.
    Why: The chain-friendly cast helper should compose with other DFx methods.
    Fails when: The method returns a plain DataFrame or skips existing columns.
    """
    _patch_spark_functions(monkeypatch)
    df = FakeDataFrame(["amount"])

    result = DFx(df).cast_if_exists("amount", "double")

    assert isinstance(result, DFx)
    assert result.unwrap().calls[-1][0:2] == ("withColumn", "amount")


def test_flatten_projects_nested_struct_fields(monkeypatch):
    """
    What: Verifies flatten recursively projects nested struct fields with aliases.
    Why: Nested source records need predictable top-level output columns.
    Fails when: Struct leaves are missed or aliases do not include parent paths.
    """
    _patch_spark_functions(monkeypatch)
    schema = T.StructType(
        [
            T.StructField("id", T.IntegerType()),
            T.StructField(
                "customer",
                T.StructType(
                    [
                        T.StructField("name", T.StringType()),
                        T.StructField(
                            "address",
                            T.StructType([T.StructField("city", T.StringType())]),
                        ),
                    ]
                ),
            ),
        ]
    )
    df = FakeDataFrame(["id", "customer"], schema=schema)

    result = DFx(df).flatten(separator="_")

    assert result.unwrap().columns == ["id", "customer_name", "customer_address_city"]


def test_with_column_proxy_preserves_chaining():
    """
    What: Verifies withColumn delegates to the wrapped DataFrame and returns DFx.
    Why: Common Spark transformations should remain chainable inside DFx.
    Fails when: The proxy returns a plain DataFrame or does not call withColumn.
    """
    df = FakeDataFrame(["id"])

    result = DFx(df).withColumn("status", FakeExpr("expr"))

    assert isinstance(result, DFx)
    assert result.unwrap().calls[-1][0:2] == ("withColumn", "status")


def test_select_proxy_preserves_chaining():
    """
    What: Verifies select delegates to the wrapped DataFrame and returns DFx.
    Why: Projection is a core Spark operation used in fluent ETL chains.
    Fails when: The proxy returns a plain DataFrame or selected columns are not forwarded.
    """
    df = FakeDataFrame(["id", "status"])

    result = DFx(df).select("id")

    assert isinstance(result, DFx)
    assert result.unwrap().calls[-1][0] == "select"


def test_filter_proxy_preserves_chaining():
    """
    What: Verifies filter delegates to the wrapped DataFrame and returns DFx.
    Why: Row filtering should compose with utility transformations.
    Fails when: The proxy returns a plain DataFrame or drops filter arguments.
    """
    df = FakeDataFrame(["id"])

    result = DFx(df).filter("id is not null")

    assert isinstance(result, DFx)
    assert result.unwrap().calls[-1][0] == "filter"


def test_where_proxy_preserves_chaining():
    """
    What: Verifies where delegates to the wrapped DataFrame and returns DFx.
    Why: Users expect Spark's filter alias to work in the wrapper.
    Fails when: The proxy returns a plain DataFrame or calls the wrong method.
    """
    df = FakeDataFrame(["id"])

    result = DFx(df).where("id is not null")

    assert isinstance(result, DFx)
    assert result.unwrap().calls[-1][0] == "where"


def test_drop_proxy_preserves_chaining():
    """
    What: Verifies drop delegates to the wrapped DataFrame and returns DFx.
    Why: Removing working columns should be possible without unwrapping.
    Fails when: The proxy returns a plain DataFrame or columns are not removed.
    """
    df = FakeDataFrame(["id", "debug"])

    result = DFx(df).drop("debug")

    assert isinstance(result, DFx)
    assert result.unwrap().columns == ["id"]
    assert result.unwrap().calls[-1][0] == "drop"


def test_with_column_renamed_proxy_preserves_chaining():
    """
    What: Verifies withColumnRenamed delegates and returns DFx.
    Why: Rename operations should be available in fluent DFx chains.
    Fails when: The proxy returns a plain DataFrame or does not rename the column.
    """
    df = FakeDataFrame(["acct_id"])

    result = DFx(df).withColumnRenamed("acct_id", "account_id")

    assert isinstance(result, DFx)
    assert result.unwrap().columns == ["account_id"]
    assert result.unwrap().calls[-1][0] == "withColumnRenamed"
