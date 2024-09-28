from sqltree import sqltree
from sqltree.dialect import Dialect, Vendor
from typing import Self, Set


class Payload:
    '''
    Wrapper around a payload with syntax checking functionality and
    string-based equality/list contains comparisons.
    '''

    __payload: str
    __tokens: Set[str]
    
    
    @property
    def tokens(self):
        return self.__tokens
    
    
    @property
    def is_syntax_correct(self):
        '''
        Attempts SQL syntax checks on payload by attaching to dummy queries.
        
        Returns true if a combined query does not throw an AST error.
        '''
        
        payload = str(self)

        return self.__is_query_syntax_correct(
            f'SELECT x FROM y WHERE z=\'{payload}\''
        ) or self.__is_query_syntax_correct(
            f'SELECT x FROM y WHERE z={payload}'
        )
    

    def __init__(
        self,
        payload: str,
        payload_tokens: Set[str],
    ) -> None:
        self.__payload = payload
        self.__tokens = payload_tokens


    def __is_query_syntax_correct(self, query: str):
        try:
            sqltree(query, dialect=Dialect(Vendor.ansi))
            return True
        except:
            return False


    def __str__(self):
        return self.__payload


    # Hash must be overriden if equality also is. Outlined in the Python
    # documentation:
    # https://docs.python.org/2/reference/datamodel.html#object.__hash__
    def __hash__(self):
        return hash(str(self))


    def __eq__(self, other: Self):
        return str(self) == str(other)