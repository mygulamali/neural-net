# neural-net

A toy neural net in C.

## Requirements

The following libraries need to be installed on your system before use.

* [GNU Scientific Library][gsl] (GSL) for the vector and matrix types, and the
  linear algebra functions.
* [CMocka][cmocka] for unit testing the code.

They can be installed on a Fedora Linux workstation (ie. my current system) with,

```bash
sudo dnf install gsl gsl-devel libcmocka libcmocka-devel
```

## License

This repository is licensed under the terms and conditions of the
[MIT License][mit_license]. Please see the `LICENSE` file for more details.

[cmocka]: https://cmocka.org/ "CMocka"
[gsl]: https://www.gnu.org/software/gsl/ "GNU Scientific Library"
[mit_license]: http://opensource.org/licenses/mit-license.php "MIT License"
