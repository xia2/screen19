import unittest

class UnitsTests(unittest.TestCase):

  def test_arithmetics(self):
    actual = 1 + 1

    expected = 2

    self.assertEqual(actual, expected)

if __name__ == '__main__':
  unittest.main()
