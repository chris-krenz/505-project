## About

This simple program is a living template for Python projects.


## Install

### Dependencies

```console
pip install -r requirements.txt
```

## Usage

### Unit Tests
```console
pytest
```

### Coverage
```console
coverage run -m pytest
coverage [report | html]
```

## Examples

### Unit Tests

#### All tests passed...
```console
=================== test session starts ===================
platform win32 -- Python 3.10.7, pytest-7.4.4, pluggy-1.4.0
rootdir: <rootdir>
plugins: anyio-3.6.1, dash-2.9.1
collected 1 item

unit_tests\test_matmul.py .                          [100%]

==================== 1 passed in 0.61s ====================

```

#### Test failed...

```console
=================== test session starts ===================
platform win32 -- Python 3.10.7, pytest-7.4.4, pluggy-1.4.0
rootdir: <rootdir>
plugins: anyio-3.6.1, dash-2.9.1
collected 1 item

unit_tests\test_matmul.py F                          [100%]

======================== FAILURES =========================
_______________________ test_matmul _______________________

    def test_matmul():

        # VALID (SUNNY) TESTS
        assert matmul(np.array([1, 7]), np.array([1, 3])) == 22  # array inputs
        assert matmul((1, 7), np.array([1, 3])) == 22            # tuple input
        assert matmul([1, 7], np.array([1, 3])) == 22            # list input

        assert matmul([1, 7], [1, 3]) == 22                 # int vals
        assert matmul([1.0, 7.0], [1, 3]) == 22.0           # float vals
        assert matmul([0, 0], [1, 3]) == 0                  # zero vals
>       assert matmul([1, -7], [-1, 3]) == -21              # neg vals
E       assert -22 == -21
E        +  where -22 = matmul([1, -7], [-1, 3])

unit_tests\test_matmul.py:26: AssertionError
================= short test summary info =================
FAILED unit_tests/test_matmul.py::test_matmul - assert -22 == -21
==================== 1 failed in 1.12s ====================
```

### Coverage
```console
Name                        Stmts   Miss  Cover
-----------------------------------------------
funcs\matmul.py                 8      1    88%
unit_tests\test_matmul.py      22      0   100%
-----------------------------------------------
TOTAL                          30      1    97%
```

## Contributors

Chris Krenz


## License

[MIT License](LICENSE)
