from __future__ import division
import datetime
import iotbx.cif.model
import os
import re

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

def get_versions():
  from dials.util.version import dials_version
  from i19.util.version import i19_version
  import xia2.XIA2Version
  versions = {
    'xia2': xia2.XIA2Version.Version,
    'dials': dials_version(),
    'i19': i19_version(),
    'aimless': 'AIMLESS, CCP4' }
  with open(find_aimless_log(), 'r') as aimlesslog:
    pattern = re.compile(" +#+ *CCP4.*#+")
    for line in aimlesslog:
      if pattern.search(line):
        versions['aimless'] = re.sub('\s\s+', ', ', line.strip("\t\n #"))
        break
  return versions

def write_cif(filename, absmin, absmax, prefix='abscorr'):
  versions = get_versions()
  block = iotbx.cif.model.block()
  block["_audit_creation_method"] = versions['xia2']
  block["_audit_creation_date"] = datetime.date.today().isoformat()

  block["_publ_section_references"] = '''
Evans, P. R. and Murshudov, G. N. (2013) Acta Cryst. D69, 1204-1214.
Winter, G. (2010) Journal of Applied Crystallography 43, 186-190.
'''

  block["_exptl_absorpt_correction_T_min"] = absmin
  block["_exptl_absorpt_correction_T_max"] = absmax
  block["_exptl_absorpt_correction_type"] = "empirical"
  block["_exptl_absorpt_process_details"] = '''
{aimless}
Scaling & analysis of unmerged intensities, absorption correction using spherical harmonics

Run via {xia2}, {dials}, {i19}
'''.format(**versions)

  cif = iotbx.cif.model.cif()
  cif[prefix] = block
  with open(filename, 'w') as fh:
    cif.show(out=fh)

def main():
  print "Generating absorption surface"
  log = find_aimless_log()

  from xia2.Toolkit.AimlessSurface import evaluate_1degree, scrape_coefficients, generate_map
  absmap = evaluate_1degree(scrape_coefficients(log))
  assert absmap.max() - absmap.min() > 0.000001, "Cannot create absorption surface: map is too flat (min: %f, max: %f)" % (absmap.min(), absmap.max())
  write_cif('absorption_surface.cif_xia2', absmap.min(), absmap.max())
  generate_map(absmap, 'absorption_surface.png')

if __name__ == '__main__':
  main()
