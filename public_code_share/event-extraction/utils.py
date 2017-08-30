import datetime

def neighborhood_long(iterable):
    """PULL NEIGHBORHOOD AROUND EACH SENTENCE: TWO SENTENCES TO EITHER SIDE"""
    iterator = iter(iterable)
    second_prev_item = None
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    next_item = next(iterator)
    for second_next_item in iterator:
        yield (second_prev_item, prev_item, current_item, next_item, second_next_item)
        second_prev_item = prev_item
        prev_item = current_item
        current_item = next_item
        next_item = second_next_item
    yield (second_prev_item, prev_item, current_item, next_item, None)
    
def neighborhood_short(iterable):
    """PULL NEIGHBORHOOD AROUND EACH SENTENCE: ONE SENTENCE TO EITHER SIDE"""
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)
    
epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    """GET UNIX TIME OF DATETIME.DATETIME() FOR WEBHOSE API"""
    return (dt - epoch).total_seconds() * 1000.0

int(unix_time_millis(datetime.datetime(2017,7,24,0,0,0)))

def pretty(d, indent=0):
    """SIMPLE PRETTY PRINT OF DICTIONARY OUTPUT"""
    for key, value in d.items():
        print(str(key) + ':\t' + str(value),'\n')
