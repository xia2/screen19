Screen19 0.213 (2022-09-29)
===========================

Bugfixes
--------

- Fix a case of screen19 failing when trying to refine the different possible Bravais settings of the found lattice.
  This was happening because screen19 was failing to filter out reflections with zero variance in the position of their observed centroids before performing the refinement. (`#53 <https://github.com/xia2/screen19/issues/53>`_)
- Fix broken help messages for ``screen19`` and ``screen19.minimum_exposure``. (`#58 <https://github.com/xia2/screen19/issues/58>`_)


Deprecations and Removals
-------------------------

- screen19 no longer supports DIALS 2.2 or Python 2.7.  To install screen19, it is recommended that you start by `installing DIALS <https://dials.github.io/installation.html>`_ version 3.4 or greater. (`#52 <https://github.com/xia2/screen19/issues/52>`_)


Misc
----

- `#43 <https://github.com/xia2/screen19/issues/43>`_, `#44 <https://github.com/xia2/screen19/issues/44>`_, `#46 <https://github.com/xia2/screen19/issues/46>`_, `#48 <https://github.com/xia2/screen19/issues/48>`_, `#50 <https://github.com/xia2/screen19/issues/50>`_, `#51 <https://github.com/xia2/screen19/issues/51>`_, `#55 <https://github.com/xia2/screen19/issues/55>`_


screen19 0.212 (2020-10-12)
===========================

Update screen19 to work with upstream API changes in `python-procrunner <https://github.com/DiamondLightSource/python-procrunner/pull/60>`_ version 2.1.x.

Misc
----

- `#37 <https://github.com/xia2/screen19/issues/37>`_, `#38 <https://github.com/xia2/screen19/issues/38>`_, `#39 <https://github.com/xia2/screen19/issues/39>`_, `#40 <https://github.com/xia2/screen19/issues/40>`_, `#41 <https://github.com/xia2/screen19/issues/41>`_


screen19 0.211 (2020-08-19)
===========================

Misc
----

- The way that dials.integrate code is called from screen19 has changed to accommodate code changes in DIALS.

  (`#31 <https://github.com/xia2/screen19/issues/31>`_)


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

  (`#30 <https://github.com/xia2/screen19/issues/30>`_)


screen19 0.208 (2020-06-25)
===========================

Features
--------

- screen19 now supports DIALS 3 and DIALS 2.2.

  `DIALS is available here <https://dials.github.io/installation.html>`_.

  * DIALS 3 is the latest major release and is actively supported.  It is currently Python 3.6 only.
  * DIALS 2.2 is a long-term support release.  It only receives bug fixes and support will be withdrawn at the end of 2020.  It supports Python 2.7 and 3.6.
  * DIALS 1.14 is no longer under active development and screen19 support for it has been withdrawn. (`#25 <https://github.com/xia2/screen19/issues/25>`_)


Bugfixes
--------

- Perform French-Wilson scaling on the integrated intensities before performing the Wilson plot analysis.
  This fixes screen19 failures for certain cases where the data consist of few reflections. (`#29 <https://github.com/xia2/screen19/issues/29>`_)
