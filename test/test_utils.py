import unittest

import mlsurvey as mls


class TestUtils(unittest.TestCase):

    def test_md5_file(self):
        md5 = mls.Utils.md5_file('files/test_md5.txt')
        self.assertEqual('70a4b9f4707d258f559f91615297a3ec', md5)
