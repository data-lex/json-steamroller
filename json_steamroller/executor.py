import json
import polars as pl
import re
from cytoolz import filter, partial, reduce
from cytoolz.dicttoolz import dissoc
from cytoolz.functoolz import thread_first
from cytoolz.itertoolz import mapcat
from polars import DataFrame
from typedefs import GenList, JsonDict, JsonList, JsonTuple, String, StrList
from typing import Any, Generator, Iterator, Optional


def write_parquet(file_name: String) -> None:
    df: DataFrame = pl.read_ndjson(f'./{file_name}.jsonl')
    df_str: DataFrame = df.select(pl.col('*').cast(str))
    df_str.write_parquet(f'./{file_name}.parquet')
    return


def write_json(file_name: String, result_list: StrList) -> None:
    with open(f'./{file_name}.jsonl', 'w') as f:
        for element in result_list:
            f.write(f'{element}\n')
    return


def is_array(record: JsonTuple) -> Optional[JsonTuple]:
    if type(record[1]) is list:
        return record


def replace_char(key_name, expression) -> String:
    return re.sub(expression, '', key_name)


def clean_key(key_name: String) -> String:
    if key_name == '_id_$oid':
        new_key: String = thread_first(
            key_name,
            (replace_char, '^\\_'),
            (replace_char, '\\$'),
            str.lower
        )
    elif '$' in key_name:
        new_key: String = thread_first(
            key_name,
            (replace_char, '\\b\\$\\w+'),
            (replace_char, '_$'),
            str.lower
        )
    else:
        new_key: String = str.lower(key_name)
    return new_key


def put_prefix(previous_key: Optional[String]) -> String:
    if previous_key is None:
        prefix: String = ''
    else:
        prefix: String = f'{previous_key}_'
    return prefix


def unnest_data(prefix: Optional[String], key_value: JsonTuple) -> Iterator:
    key: String = f'{put_prefix(prefix)}{key_value[0]}'
    if key_value[1] is None:
        pass
    elif type(key_value[1]) == dict:
        yield from mapcat(partial(unnest_data, key), key_value[1].items())
    else:
        new_key: String = clean_key(key)
        value: Any = key_value[1]
        yield {new_key: value}


def process_record(json_record: JsonDict) -> Generator:
    pairs: Iterator = mapcat(partial(unnest_data, None), json_record.items())
    record: JsonDict = reduce(lambda x, y: x | y, pairs)
    arrays: JsonDict = dict(filter(is_array, record.items()))
    if arrays.keys():
        flats: JsonDict = reduce(dissoc, arrays.keys(), record)
        exploded: JsonList = [flats | {k: i} for k in arrays.keys() for i in arrays[k]]
        yield from mapcat(process_record, exploded)
    else:
        yield record


def call_processing(raw_record: String) -> Generator:
    json_record: JsonDict = json.loads(raw_record)
    return process_record(json_record)


def read_file(path: String) -> StrList:
    with open(path, 'r') as f:
        file: StrList = f.readlines()
    return file


def main() -> None:
    path: String = input('Dataset Path: ')
    raw_file: StrList = read_file(path)
    result_gen: GenList = [call_processing(x) for x in raw_file]
    result_str: StrList = [json.dumps(y) for x in result_gen for y in x]
    write_json(path, result_str)
    write_parquet(path)
    return


if __name__ == '__main__':
    main()
