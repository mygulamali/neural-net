# neural-net

A toy neural net in C.

## Requirements

The following libraries and applications need to be installed on your system
before use.

* [GNU Scientific Library][gsl] (GSL) for the vector and matrix types, and the
  linear algebra functions.
* [CMocka][cmocka] to unit test the code.
* [Valgrind][valgrind] to detect memory leaks.

They can be installed on a Fedora Linux workstation (ie. my current system)
with,

```bash
sudo dnf install -y gsl gsl-devel libcmocka libcmocka-devel valgrind
```

## Testing

The test suite can be compiled and run with the rule,

```bash
make tests
```

The library functions can be checked for memory leaks using the rule,

```bash
make mem_tests
```

## License

This repository is licensed under the terms and conditions of the
[MIT License][mit_license]. Please see the `LICENSE` file for more details.

[cmocka]: https://cmocka.org/ "CMocka"
[gsl]: https://www.gnu.org/software/gsl/ "GNU Scientific Library"
[mit_license]: http://opensource.org/licenses/mit-license.php "MIT License"
[valgrind]: https://valgrind.org/ "Valgrind"
