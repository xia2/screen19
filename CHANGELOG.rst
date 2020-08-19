screen19 0.211 (2020-08-19)
===========================

Misc
----

- The way that dials.integrate code is called from screen19 has changed to accommodate code changes in DIALS.

  (#31)


screen19 0.210 (2020-06-26)
===========================

Bugfixes
--------

- Add missing files to source release

screen19 0.209 (2020-06-25)
===========================

Bugfixes
--------

- Some overly verbose log messages from the French-Wilson calculation are now suppressed by default.
  You can still see them, if you want to, by running screen19 in debug mode with::

      screen19 verbosity=1 <other arguments>

  (#30)


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
