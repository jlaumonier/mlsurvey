import unittest

import mlsurvey as mls


class TestVisualization(unittest.TestCase):
    """Cannot do more test at the moment"""

    def test_init_visualization_should_be_initialized(self):
        v = mls.Visualization()
        self.assertIsNotNone(v)
