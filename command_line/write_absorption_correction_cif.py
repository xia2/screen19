from __future__ import division
import os

def find_scale_dir():
  assert os.path.exists('xia2.json')
  from xia2.Schema.XProject import XProject
  xinfo = XProject.from_json(filename='xia2.json')
  crystals = xinfo.get_crystals()
  assert len(crystals) == 1
  crystal = next(crystals.itervalues())
  return os.path.join(crystal.get_name(), 'scale')

def find_aimless_log():
  scaledir = os.path.join('DEFAULT', 'scale') # best guess
  if not os.path.exists(scaledir):
    scaledir = find_scale_dir()
  logs = [f for f in os.listdir(scaledir)
      if f.endswith('_aimless.log') and os.path.isfile(os.path.join(scaledir, f))]
  logs.sort(key=lambda x: int(x.split('_')[0]))
  lastlog = os.path.join(scaledir, logs[-1])
  assert os.path.isfile(lastlog)
  return lastlog

def main(log, png):
  from xia2.Toolkit.AimlessSurface import evaluate_1degree, scrape_coefficients
  evaluate_1degree(scrape_coefficients(log), png)

if __name__ == '__main__':
  print "Generating absorption surface map"
  main(find_aimless_log(), 'absorption_surface.png')
