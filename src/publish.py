#!/usr/bin/env python
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cProfile
import datetime
import shutil
import traceback
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pstats import Stats
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, TextIO

from pandas import DataFrame

from lib.concurrent import thread_map
from lib.constants import EXCLUDE_FROM_MAIN_TABLE, OUTPUT_COLUMN_ADAPTER, SRC
from lib.error_logger import ErrorLogger
from lib.io import display_progress, export_csv, pbar, read_file, read_lines
from lib.memory_efficient import (
    convert_csv_to_json_records,
    get_table_columns,
    table_breakout,
    table_cross_product,
    table_drop_nan_columns,
    table_group_tail,
    table_join,
    table_read_column,
    table_rename,
    table_sort,
)
from lib.pipeline_tools import get_schema
from lib.sql import create_sqlite_database, table_as_csv, table_import_from_file, table_merge
from lib.time import date_range


def _subset_grouped_key(
    main_table_path: Path, output_folder: Path, desc: str = None
) -> Iterable[Path]:
    """ Outputs a subsets of the table with only records with a particular key """

    # Read the header of the main file to get the columns
    with open(main_table_path, "r") as fd:
        header = next(fd)

    # Do a first sweep to get the number of keys so we can accurately report progress
    key_set = set()
    for line in read_lines(main_table_path, skip_empty=True):
        key, data = line.split(",", 1)
        key_set.add(key)

    # We make use of the main table being sorted by <key, date> and do a linear sweep of the file
    # assuming that once the key changes we won't see it again in future lines
    key_folder: Path = None
    current_key: str = None
    file_handle: TextIO = None
    progress_bar = pbar(total=len(key_set), desc=desc)
    for idx, line in enumerate(read_lines(main_table_path, skip_empty=True)):
        key, data = line.split(",", 1)

        # Skip the header line
        if idx == 0:
            continue

        # When the key changes, close the previous file handle and open a new one
        if current_key != key:
            if file_handle:
                file_handle.close()
            if key_folder:
                yield key_folder / "main.csv"
            current_key = key
            key_folder = output_folder / key
            key_folder.mkdir(exist_ok=True)
            file_handle = (key_folder / "main.csv").open("w")
            file_handle.write(f"{header}")
            progress_bar.update(1)

        file_handle.write(f"{key},{data}")

    # Close the last file handle and we are done
    file_handle.close()
    progress_bar.close()


def copy_tables(tables_folder: Path, public_folder: Path) -> None:
    """
    Copy tables as-is from the tables folder into the public folder.

    Arguments:
        tables_folder: Input folder where all CSV files exist.
        public_folder: Output folder where the CSV files will be copied to.
    """
    for output_file in pbar([*tables_folder.glob("*.csv")], desc="Copy tables"):
        shutil.copy(output_file, public_folder / output_file.name)


def make_main_table(
    tables_folder: Path, output_path: Path, logger: ErrorLogger = ErrorLogger()
) -> None:
    """
    Build a flat view of all tables combined, joined by <key> or <key, date>.
    Arguments:
        tables_folder: Input folder where all CSV files exist.
    """

    # Use a temporary directory for intermediate files
    with TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        # Merge all output files into a single table
        keys_table_path = workdir / "keys.csv"
        keys_table = read_file(tables_folder / "index.csv", usecols=["key"])
        export_csv(keys_table, keys_table_path)
        logger.log_info("Created keys table")

        # Add a date to each region from index to allow iterative left joins
        max_date = (datetime.datetime.now() + datetime.timedelta(days=1)).date().isoformat()
        date_list = date_range("2020-01-01", max_date)
        date_table_path = workdir / "dates.csv"
        export_csv(DataFrame(date_list, columns=["date"]), date_table_path)
        logger.log_info("Created dates table")

        # Create a temporary working table file which can be used during the steps
        temp_file_path = workdir / "main.tmp.csv"
        table_cross_product(keys_table_path, date_table_path, temp_file_path)
        logger.log_info("Created cross product table")

        # Add all the index columns to seed the main table
        main_table_path = workdir / "main.csv"
        table_join(
            temp_file_path, tables_folder / "index.csv", ["key"], main_table_path, how="outer"
        )
        logger.log_info("Joined with table index")

        non_dated_columns = set(get_table_columns(main_table_path))
        for table_file_path in pbar([*tables_folder.glob("*.csv")], desc="Make main table"):
            table_name = table_file_path.stem
            if table_name not in EXCLUDE_FROM_MAIN_TABLE:

                table_columns = get_table_columns(table_file_path)
                if "date" in table_columns:
                    join_on = ["key", "date"]
                else:
                    join_on = ["key"]

                    # Keep track of columns which are not indexed by date
                    non_dated_columns = non_dated_columns | set(table_columns)

                # Iteratively perform left outer joins on all tables
                table_join(main_table_path, table_file_path, join_on, temp_file_path, how="outer")
                shutil.move(temp_file_path, main_table_path)
                logger.log_info(f"Joined with table {table_name}")

        # Drop rows with null date or without a single dated record
        # TODO: figure out a memory-efficient way to do this

        # Ensure that the table is appropriately sorted ans write to output location
        table_sort(main_table_path, output_path)
        logger.log_info("Sorted main table")


def _make_location_key_and_date_table(tables_folder: Path, output_path: Path) -> None:
    # Use a temporary directory for intermediate files
    with TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        # Make sure that there is an index table present
        index_table = tables_folder / "index.csv"
        assert index_table.exists(), "Index table not found"

        # Create a single-column table with only the keys
        keys_table_path = workdir / "location_keys.csv"
        with open(keys_table_path, "w") as fd:
            fd.write(f"location_key\n")
            fd.writelines(f"{value}\n" for value in table_read_column(index_table, "location_key"))

        # Add a date to each region from index to allow iterative left joins
        max_date = (datetime.datetime.now() + datetime.timedelta(days=1)).date().isoformat()
        date_table_path = workdir / "dates.csv"
        with open(date_table_path, "w") as fd:
            fd.write("date\n")
            fd.writelines(f"{value}\n" for value in date_range("2020-01-01", max_date))

        # Output all combinations of <key x date>
        table_cross_product(keys_table_path, date_table_path, output_path)


def make_main_table_v3(
    tables_folder: Path, output_path: Path, drop_empty_columns: bool = False
) -> None:
    """
    Build a flat view of all tables combined, joined by <key> or <key, date>.
    Arguments:
        tables_folder: Input directory where all CSV files exist.
        output_path: Output directory for the resulting main.csv file.
    """
    # Use a temporary directory for intermediate files
    with TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        # Use temporary files to avoid computing everything in memory
        temp_input = workdir / "tmp.1.csv"
        temp_output = workdir / "tmp.2.csv"

        # Start with all combinations of <location_key x date>
        _make_location_key_and_date_table(tables_folder, temp_output)
        temp_input, temp_output = temp_output, temp_input

        for table_file_path in tables_folder.glob("*.csv"):
            table_name = table_file_path.stem

            # Procede depending on whether this table should be excluded
            if table_name in ("main",):
                continue

            table_columns = get_table_columns(table_file_path)
            if "date" in table_columns:
                join_on = ["location_key", "date"]
            else:
                join_on = ["location_key"]

            # Iteratively perform left outer joins on all tables
            table_join(temp_input, table_file_path, join_on, temp_output, how="outer")

            # Flip-flop the temp files to avoid a copy
            temp_input, temp_output = temp_output, temp_input

        # Drop rows with null date or without a single dated record
        # TODO: figure out a memory-efficient way to do this

        # Remove columns which provide no data because they are only null values
        if drop_empty_columns:
            table_drop_nan_columns(temp_input, temp_output)
            temp_input, temp_output = temp_output, temp_input

        # Ensure that the table is appropriately sorted and write to output location
        table_sort(temp_input, output_path)


def make_main_table_sqlite(
    tables_folder: Path, output_path: Path, drop_empty_columns: bool = False
) -> None:
    """
    Build a flat view of all tables combined, joined by <key> or <key, date>.

    Arguments:
        tables_folder: Input directory where all CSV files exist.
        output_path: Output directory for the resulting main.csv file.
        location_key: Name of the key to use for the location, "key" (v2) or "location_key" (v3).
        exclude_tables: List of tables that should be excluded from the joint output.
        drop_empty_columns: Flag indicating whether columns with no data should be dropped.
    """
    # Use a temporary directory for intermediate files
    with TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        # Start with all combinations of <key x date>
        cross_product_table_path = workdir / "cross-product.csv"
        _make_location_key_and_date_table(tables_folder, cross_product_table_path)

        # Import all tables into a temporary database, default to using in-memory db
        with create_sqlite_database() as conn:

            cross_product_schema = {"location_key": str, "date": str}
            table_import_from_file(conn, cross_product_table_path, schema=cross_product_schema)

            # Get a list of all tables indexed by <location_key> or by <location_key, date>
            schema = get_schema()
            idx1 = []
            idx2 = [cross_product_table_path.stem]
            for table_file_path in tables_folder.glob("*.csv"):
                table_name = table_file_path.stem
                table_columns = get_table_columns(table_file_path)
                table_schema = {col: schema.get(col, str) for col in table_columns}
                table_import_from_file(
                    conn, table_file_path, table_name=table_name, schema=table_schema
                )
                if "date" in table_columns:
                    idx2.append(table_name)
                else:
                    idx1.append(table_name)

            # More readable variable name
            key = "location_key"

            # All joins are LEFT OUTER JOIN
            join_method = "LEFT OUTER"

            # Perform the join in 3 steps, first join all tables by key
            table_merge(conn, idx1, on=[key], how=join_method, into_table="_idx1")

            # Next, join all the tables by <key, date>
            table_merge(conn, idx2, on=[key, "date"], how=join_method, into_table="_idx2")

            # Finally join the two sets of tables together
            # NOTE: order of tables merged matters, since SQLite only supports *LEFT* OUTER joins
            table_merge(conn, ["_idx2", "_idx1"], on=[key], how=join_method, into_table="_main")

            # TODO: avoid use of temporary files for this
            temp_input = workdir / "tmp.1.csv"
            temp_output = workdir / "tmp.2.csv"

            # Dump the table as a CSV file
            table_as_csv(conn, "_main", temp_output, sort_by=[key, "date"])
            temp_input, temp_output = temp_output, temp_input

            # Drop rows with null date or without a single dated record
            # TODO: figure out a memory-efficient way to do this

            # Remove columns which provide no data because they are only null values
            if drop_empty_columns:
                table_drop_nan_columns(temp_input, temp_output)
                temp_input, temp_output = temp_output, temp_input

            # Write to output location
            shutil.copy(temp_input, output_path)


def create_table_subsets(main_table_path: Path, output_path: Path) -> Iterable[Path]:

    latest_path = output_path / "latest"
    latest_path.mkdir(parents=True, exist_ok=True)

    def subset_latest(csv_file: Path) -> Path:
        output_file = latest_path / csv_file.name
        table_group_tail(csv_file, output_file)
        return output_file

    # Create a subset with the latest known day of data for each key
    map_func = subset_latest
    yield from thread_map(map_func, [*output_path.glob("*.csv")], desc="Latest subset")

    # Create subsets with each known key
    yield from _subset_grouped_key(main_table_path, output_path, desc="Grouped key subsets")


def convert_tables_to_json(
    csv_folder: Path, output_folder: Path, logger: ErrorLogger = ErrorLogger()
) -> Iterable[Path]:
    def try_json_covert(schema: Dict[str, str], csv_file: Path) -> Path:
        # JSON output defaults to same as the CSV file but with extension swapped
        json_output = output_folder / str(csv_file.relative_to(csv_folder)).replace(".csv", ".json")
        json_output.parent.mkdir(parents=True, exist_ok=True)

        # Converting to JSON is not critical and it may fail in some corner cases
        # As long as the "important" JSON files are created, this should be OK
        try:
            logger.log_debug(f"Converting {csv_file} to JSON")
            convert_csv_to_json_records(schema, csv_file, json_output)
            return json_output
        except Exception as exc:
            error_message = f"Unable to convert CSV file {csv_file} to JSON"
            logger.log_error(error_message, traceback=traceback.format_exc())
            return None

    # Convert all CSV files to JSON using values format
    map_iter = list(csv_folder.glob("**/*.csv"))
    map_func = partial(try_json_covert, get_schema())
    for json_output in thread_map(map_func, map_iter, max_workers=2, desc="JSON conversion"):
        if json_output is not None:
            yield json_output


def publish_location_breakouts(tables_folder: Path, output_folder: Path) -> None:
    """
    Breaks out each of the tables in `tables_folder` based on location key, and writes them into
    subdirectories of `output_folder`. This method also joins *all* the tables into a main.csv
    table, which will be more comprehensive than the global main.csv.

    Arguments:
        tables_folder: Directory containing input CSV files.
        output_folder: Output path for the resulting location data.
    """
    # Break out each table into separate folders based on the location key
    for csv_path in tables_folder.glob("*.csv"):
        print(f"Breaking out table {csv_path.name}")
        table_breakout(csv_path, output_folder, "location_key")


def publish_location_aggregates(
    tables_folder: Path, output_folder: Path, location_keys: Iterable[str]
) -> None:
    """
    This method joins *all* the tables for each location into a main.csv table.

    Arguments:
        tables_folder: Directory containing input CSV files.
        output_folder: Output path for the resulting location data.
        location_keys: List of location keys to do aggregation for.
    """
    # Create a main.csv file for each of the locations in parallel
    map_func = lambda key: make_main_table_v3(
        tables_folder / key, output_folder / key / "main.csv", drop_empty_columns=True
    )
    list(thread_map(map_func, list(location_keys), desc="Creating location subsets"))


def publish_global_tables(tables_folder: Path, output_folder: Path) -> None:
    """
    Copy all the tables from `tables_folder` into `output_folder` converting the column names to the
    latest schema, and join all the tables into a single main.csv file.

    Arguments:
        tables_folder: Input directory containing tables as CSV files.
        output_folder: Directory where the output tables will be written.
    """
    with TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        for csv_path in tables_folder.glob("*.csv"):
            # Copy all output files to a temporary folder, renaming columns if necessary
            print(f"Renaming columns for {csv_path.name}")
            table_rename(csv_path, workdir / csv_path.name, OUTPUT_COLUMN_ADAPTER)

        for csv_path in tables_folder.glob("*.csv"):
            # Sort output files by location key, since the following breakout step requires it
            print(f"Sorting {csv_path.name}")
            table_sort(workdir / csv_path.name, output_folder / csv_path.name, ["location_key"])

    # Main table is not created for the global output, because it's too big to be useful


def main_v3(output_folder: Path, tables_folder: Path, show_progress: bool = True) -> None:
    """
    This script takes the processed outputs located in `tables_folder` and publishes them into the
    output folder by performing the following operations:

        1. Copy all the tables from `tables_folder` to `output_folder`, renaming fields if
           necessary.
        2. Create different slices of data, such as the latest known record for each region, files
           for the last N days of data, files for each individual region.
        3. Produce a main table, created by iteratively performing left outer joins on all other
           tables for each slice of data (bot not for the global tables).
    """
    with display_progress(show_progress):

        # Wipe the output folder first
        for item in output_folder.glob("*"):
            if item.name.startswith("."):
                continue
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

        # Create the folder which will be published using a stable schema
        v3_folder = output_folder / "v3"
        v3_folder.mkdir(exist_ok=True, parents=True)

        # Publish the tables containing all location keys
        publish_global_tables(tables_folder, v3_folder)

        # Break out each table into separate folders based on the location key
        publish_location_breakouts(v3_folder, v3_folder)

        # Aggregate the independent tables for each location
        location_keys = table_read_column(v3_folder / "index.csv", "location_key")
        publish_location_aggregates(v3_folder, v3_folder, location_keys)

        # Convert all CSV files to JSON using values format
        # convert_tables_to_json([*v3_folder.glob("**/*.csv")], v3_folder)


def main_v2(output_folder: Path, tables_folder: Path, show_progress: bool = True) -> None:
    """
    This script takes the processed outputs located in `tables_folder` and publishes them into the
    output folder by performing the following operations:

        1. Copy all the tables as-is from `tables_folder` to `output_folder`
        2. Produce a main table, created by iteratively performing left outer joins on all other
           tables (with a few exceptions)
        3. Create different slices of data, such as the latest known record for each region, files
           for the last N days of data, files for each individual region
    """
    with display_progress(show_progress):

        # Wipe the output folder first
        for item in output_folder.glob("*"):
            if item.name.startswith("."):
                continue
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

        # Create the folder which will be published using a stable schema
        v2_folder = output_folder / "v2"
        v2_folder.mkdir(exist_ok=True, parents=True)

        # Copy all output files to the V2 folder
        copy_tables(tables_folder, v2_folder)

        # Create the main table and write it to disk
        main_table_path = v2_folder / "main.csv"
        make_main_table(tables_folder, main_table_path)

        # Create subsets for easy API-like access to slices of data
        create_table_subsets(main_table_path, v2_folder)

        # Convert all CSV files to JSON using values format
        convert_tables_to_json([*v2_folder.glob("**/*.csv")], v2_folder)


if __name__ == "__main__":

    # Process command-line arguments
    output_root = SRC / ".." / "output"
    argparser = ArgumentParser()
    argparser.add_argument("--profile", action="store_true")
    argparser.add_argument("--no-progress", action="store_true")
    argparser.add_argument("--tables-folder", type=str, default=str(output_root / "tables"))
    argparser.add_argument("--output-folder", type=str, default=str(output_root / "public"))
    args = argparser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    main_v3(Path(args.output_folder), Path(args.tables_folder), show_progress=not args.no_progress)
    # main_v2(Path(args.output_folder), Path(args.tables_folder), show_progress=not args.no_progress)

    if args.profile:
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumtime")
        stats.print_stats(20)
