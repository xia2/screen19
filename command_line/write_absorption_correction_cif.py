from __future__ import division
import datetime
import iotbx.cif.model
import os
import xia2.XIA2Version

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

def write_cif(filename, absmin, absmax, prefix='abscorr'):
  block = iotbx.cif.model.block()
  block["_audit_creation_method"] = xia2.XIA2Version.Version
  block["_audit_creation_date"] = datetime.date.today().isoformat()

  block["_publ_section_references"] = '''
Winter, G. (2010) Journal of Applied Crystallography 43
'''

  block["_exptl_absorpt_correction_T_min"] = absmin
  block["_exptl_absorpt_correction_T_max"] = absmax

  cif = iotbx.cif.model.cif()
  cif[prefix] = block
  with open(filename, 'w') as fh:
    cif.show(out=fh)

def main():
  print "Generating absorption surface"
  log = find_aimless_log()

  from xia2.Toolkit.AimlessSurface import evaluate_1degree, scrape_coefficients, generate_map
  absmap = evaluate_1degree(scrape_coefficients(log))

  write_cif('absorption_surface.cif_xia2', absmap.min(), absmap.max())
  generate_map(absmap, 'absorption_surface.png')

if __name__ == '__main__':
  main()
