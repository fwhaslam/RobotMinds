#
#   Test Script:L ensure that all source scripts compile
#

import py_compile
import pathlib
import unittest

class test_RobotMinds(unittest.TestCase):

    def test_AllScriptsCompile(self):

        root = pathlib.Path('../../main/python')

        file_count = 0
        failure_count = 0
        for f in root.rglob('*.py'):
            file_count += 1
            path = str(f)
            print("\nChecking :: ",path)
            result = py_compile.compile(path)
            if ( result == None ): failure_count += 1
            print("Result=",result)

        print("\nReviewed",file_count,"Files")
        print("Total Failures",failure_count)

        self.assertEqual( 0, failure_count )


if __name__ == '__main__':
    unittest.main()