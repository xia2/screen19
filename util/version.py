from __future__ import absolute_import, division, print_function

# DIALS version numbers are constructed from
#  1. a common prefix
__i19_version_format = "DIALS/i19 %s"
#  2. the most recent annotated git tag (or failing that: a default string)
__i19_version_default = "0"
#  3. a dot followed by the number of commits since that tag
#  4. a dash followed by a lowercase 'g' and the current commit id

# When run from a development installation the version information is extracted
# from the git repository. Otherwise it is read from the file '.gitversion' in the
# module directory.


def i19_version():
    """Try to obtain the current git revision number
     and store a copy in .gitversion"""
    version = None

    try:
        import libtbx.load_env
        import os

        i19_path = libtbx.env.dist_path("i19")
        version_file = os.path.join(i19_path, ".gitversion")

        # 1. Try to access information in .git directory
        #    Regenerate .gitversion if possible
        if os.path.exists(os.path.join(i19_path, ".git")):
            try:
                import subprocess

                with open(os.devnull, "w") as devnull:
                    version = subprocess.check_output(
                        ["git", "describe"], cwd=i19_path, stderr=devnull
                    ).rstrip()
                    version = version.replace("-", ".", 1)
                with open(version_file, "w") as gv:
                    gv.write(version)
            except Exception:
                if version == "":
                    version = None

        # 2. If .git directory missing or 'git describe' failed, read .gitversion
        if (version is None) and os.path.exists(version_file):
            with open(version_file, "r") as gv:
                version = gv.read().rstrip()
    except Exception:  # ignore any errors, use default information instead
        pass

    if version is None:
        version = __i19_version_format % __i19_version_default
    else:
        version = __i19_version_format % version

    return version
