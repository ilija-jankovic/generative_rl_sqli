def match_list_lengths(list1: list, list2: list):
    '''
    Extends `list1` or `list2` depending on whether one is smaller
    with respect to the other, matching lengths.
    
    The added elements are a sliced tiling of the smaller list.
    '''
    
    list1_len = len(list1)
    list2_len = len(list2)
    
    if list1_len == list2_len:
        return
    
    smaller_list = list1 if list1_len < list2_len else list2
    bigger_list = list1 if list1_len > list2_len else list2
    
    while len(smaller_list) < len(bigger_list):
        smaller_list.extend(smaller_list)
    
    del smaller_list[len(bigger_list):]