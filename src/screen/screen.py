"""The main screening script."""

import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("images", type=str, help="Path to the images to process.")


def pipeline(images):
    subprocess.run(["dials.import", images])
    subprocess.run(["dials.find_spots", "imported.expt"])
    subprocess.run(["dials.index", "imported.expt", "strong.refl"])
    subprocess.run(["dev.dials.pixel_histogram", "indexed.refl"])
    subprocess.run(["dials.refine", "indexed.expt", "indexed.refl"])
    subprocess.run(["dials.integrate", "refined.expt", "refined.refl"])
    subprocess.run(["screen19.minimum_exposure", "integrated.expt", "integrated.refl"])


def main(args=None):
    args = parser.parse_args(args)
    pipeline(args.images)
