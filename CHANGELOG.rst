screen19 0.208 (2020-06-25)
===========================

Features
--------

- screen19 now supports DIALS 3 and DIALS 2.2.

  `DIALS is available here <https://dials.github.io/installation.html>`_.

  * DIALS 3 is the latest major release and is actively supported.  It is currently Python 3.6 only.
  * DIALS 2.2 is a long-term support release.  It only receives bug fixes and support will be withdrawn at the end of 2020.  It supports Python 2.7 and 3.6.
  * DIALS 1.14 is no longer under active development and screen19 support for it has been withdrawn. (#25)


Bugfixes
--------

- Perform French-Wilson scaling on the integrated intensities before performing the Wilson plot analysis.
  This fixes screen19 failures for certain cases where the data consist of few reflections. (#29)
