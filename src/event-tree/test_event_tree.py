from event_tree import EventTree


class TestEventTree:
    """Runs unit tests for the event tree class"""

    def test_increment(self):
        et = EventTree()
        x = 1
        assert et.increment(x) == 2
        