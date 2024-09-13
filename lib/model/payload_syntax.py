from sqltree import sqltree

def __is_query_syntax_correct(query: str):
    try:
        sqltree(query)
        return True
    except:
        return False
    

def is_payload_syntax_correct(payload: str):
    '''
    Attempts SQL syntax checks on payload by attaching to dummy queries.
    
    Returns true if a combined query does not throw an error.
    '''

    return __is_query_syntax_correct(f'SELECT x FROM y WHERE z=\'{payload}\'') \
        or __is_query_syntax_correct(f'SELECT x FROM y WHERE z={payload}')
