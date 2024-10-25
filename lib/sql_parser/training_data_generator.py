import random
from typing import Dict, List

from lib.sql_parser import sql_data_service, sql_template_parser
from lib.nlp.token_parser import TokenParser
from lib.util import match_list_lengths


def generate_training_data(
    schema: Dict[str, str],
    max_length: int | None, 
    token_parser: TokenParser,
) -> List[List[int]]:
    payloads = sql_data_service.load_payload_files(domain_name='localhost')

    queries = sql_data_service.load_wikisql_queries()
    queries = sql_template_parser.generate_randomised_examples(
        schema=schema,
        queries=queries,
    )

    sql_data_service.save_parsed_query_templates(queries)

    # Ensure equally distributed categories of payloads and queries
    # before uniform shuffling. Both categories are important for training,
    # so equal importance is assumed.
    #
    # Note that this will duplicate members in the smaller list, allowing
    # for repeated data.
    match_list_lengths(payloads, queries)
    
    training_data = payloads + queries

    # Uniformly shuffle across length-matched categories.
    random.shuffle(training_data)
    
    print('\n'.join(training_data[:100]))
    print('\nPrinted slice of 100 rows embedding training data.')

    print('Encoding SQL fragment(s) for embeddings...')

    training_data = token_parser.parse(training_data)

    # Limit embedding training data if defined.
    training_data = training_data if max_length is None else training_data[:max_length]

    print(f'{len(training_data)} fragment(s) encoded.')
    
    return training_data
