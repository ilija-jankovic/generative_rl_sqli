import unittest

from lib.util import match_list_lengths

class TestEnvironment(unittest.TestCase):
    def test_equal_match_list_lengths(self):
        '''
        match_list_lengths returns unmodified lists if of equal length.
        '''
        
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]

        match_list_lengths(list1, list2)

        self.assertEqual(len(list1), 3)
        
        self.assertEqual(list1[0], 1)
        self.assertEqual(list1[1], 2)
        self.assertEqual(list1[2], 3)
        
        self.assertEqual(len(list2), 3)

        self.assertEqual(list2[0], 4)
        self.assertEqual(list2[1], 5)
        self.assertEqual(list2[2], 6)

    def test_list1_greater_match_list_lengths(self):
        '''
        match_list_lengths returns cut-off tiled list2 if list1 is larger.
        '''
        
        list1 = [1, 2, 3, 4, 5,]
        list2 = [6, 7,]

        match_list_lengths(list1, list2)
        
        self.assertEqual(len(list1), 5)

        self.assertEqual(list1[0], 1)
        self.assertEqual(list1[1], 2)
        self.assertEqual(list1[2], 3)
        self.assertEqual(list1[3], 4)
        self.assertEqual(list1[4], 5)
        
        self.assertEqual(len(list2), 5)
        
        self.assertEqual(list2[0], 6)
        self.assertEqual(list2[1], 7)
        self.assertEqual(list2[2], 6)
        self.assertEqual(list2[3], 7)
        self.assertEqual(list2[4], 6)
        
    def test_list2_greater_match_list_lengths(self):
        '''
        match_list_lengths returns cut-off tiled list1 if list2 is larger.
        '''
        
        list1 = [1, 2,]
        list2 = [3, 4, 5, 6, 7,]

        match_list_lengths(list1, list2)
        
        self.assertEqual(len(list1), 5)

        self.assertEqual(list1[0], 1)
        self.assertEqual(list1[1], 2)
        self.assertEqual(list1[2], 1)
        self.assertEqual(list1[3], 2)
        self.assertEqual(list1[4], 1)
        
        self.assertEqual(len(list2), 5)
        
        self.assertEqual(list2[0], 3)
        self.assertEqual(list2[1], 4)
        self.assertEqual(list2[2], 5)
        self.assertEqual(list2[3], 6)
        self.assertEqual(list2[4], 7)

if __name__ == '__main__':
    unittest.main()