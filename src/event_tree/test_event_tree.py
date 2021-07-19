#from event_tree import EventTree
import event_tree as et

def test_check_list_contains_strings() -> None:
    str_list = ['blah', 'path/to/node', '123', '~$%^']
    non_str_list = [123, 'path/to/node', True]

    assert et.check_list_contains_strings(str_list) == True
    assert et.check_list_contains_strings(non_str_list) == False

def test_check_sampling_zero_paths_param() -> None:
    """Tests the function that is checking the sampling zero paths param"""
    sz_paths = [('edge_2',),('edge_1', 'edge_3')]
    assert et.check_sampling_zero_paths_param(sz_paths) == sz_paths
    
    sz_paths = [1,2,3,4]
    assert et.check_sampling_zero_paths_param(sz_paths) == None

    sz_paths = [('path', 'to'), (123, 'something'), 'path/to']
    assert et.check_sampling_zero_paths_param(sz_paths) == None

def test_check_tuple_contains_strings() -> None:
    """Tests the function that is checking that a tuple only contains strings"""
    tup = ('string thing',)
    assert et.check_tuple_contains_strings(tup) == True
    tup = ('string one', 'string two')
    assert et.check_tuple_contains_strings(tup) == True
    tup = (1, '2')
    assert et.check_tuple_contains_strings(tup) == False    

def test_create_edges() -> None:
    assert 1 == 1