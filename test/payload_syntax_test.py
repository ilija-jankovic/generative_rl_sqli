import unittest

from lib.model.payload import Payload

class TestPayloadSyntax(unittest.TestCase):
    def test_empty_payload_is_correct_syntax(self):
        '''
        Empty payload passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_lowercase_alphabetic_payload_is_correct_syntax(self):
        '''
        Payload of lowercase alphabetic characters passes SQL
        syntax check.
        '''
        
        is_payload_valid_syntax = Payload('test').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_uppercase_alphabetic_payload_is_correct_syntax(self):
        '''
        Payload of uppercase alphabetic characters passes SQL
        syntax check.
        '''
        
        is_payload_valid_syntax = Payload('TEST').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_numeric_payload_is_correct_syntax(self):
        '''
        Payload of numeric characters passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('123').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_alphanumeric_payload_is_correct_syntax(self):
        '''
        Payload of alphanumeric characters passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('testTEST123').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_single_quote_payload_is_incorrect_syntax(self):
        '''
        Payload of a single quote does not pass SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\'').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, False)


    def test_two_single_quotes_payload_is_correct_syntax(self):
        '''
        Payload of two single quotes does passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\'\'').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_backslash_payload_is_correct_syntax(self):
        '''
        Payload of a backslash passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\\').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_single_quote_comment_payload_is_correct_syntax(self):
        '''
        Payload of a single quote and double-dash comment passes
        SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\'--').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)
        

    def test_single_quote_mysql_hash_comment_payload_is_correct_syntax(self):
        '''
        Payload of a single quote and  hash (MySQL comment) passes
        SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\'#').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_single_quote_mysql_dash_comment_payload_is_correct_syntax(self):
        '''
        Payload of a single quote, comment, then space (MySQL
        double-dash comment constraint) passes SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload('\'-- ').is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)
        
        
    def test_single_column_union_injection_payload_is_correct_syntax(self):
        '''
        Payload of a single column UNION injection passes SQL
        syntax check.
        '''
        
        is_payload_valid_syntax = Payload(
            '\' UNION SELECT x FROM y-- '
        ).is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)
        
        
    def test_three_columns_union_injection_payload_is_correct_syntax(self):
        '''
        Payload of a three column UNION injection passes SQL
        syntax check.
        '''
        
        is_payload_valid_syntax = Payload(
            '\' UNION SELECT x, y, z FROM y-- '
        ).is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, True)


    def test_incorrect_keyword_injection_payload_is_incorrect_syntax(self):
        '''
        Payload of a single column UNION injection with UNION
        misspelled as UNIONX does not pass SQL syntax check.
        '''
        
        is_payload_valid_syntax = Payload(
            '\' UNIONX SELECT x FROM y-- '
        ).is_syntax_correct
        
        self.assertEqual(is_payload_valid_syntax, False)



if __name__ == '__main__':
    unittest.main()