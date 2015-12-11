from __future__ import division

import fnmatch
import os
import re
import sys

filelist = [f for f in os.listdir('.') if os.path.isfile(f) and fnmatch.fnmatch(f, '*.cbf')]

def test_wavelength(data, straight_return=False):
  m = re.search('# Wavelength ([0-9]+\.[0-9]+) A', data)
  if m:
    wl = float(m.group(1))
    if straight_return: return wl
    if wl < 0.65 or wl > 0.95:
      raise Exception("Wavelength according to header is %f A. Is this correct?" % wl)
    return
  raise Exception("No wavelength found in header!")

def test_beampos(data):
  m = re.search('# Beam_xy \(([0-9,. ]+)\) pixels', data)
  if m:
    if m.group(1) == "0.00, 0.00":
      raise Exception("Beam position not set")
    return
  raise Exception("No beam position found in header!")

def test_distance(data):
  m = re.search('# Detector_distance ([0-9.]+) m', data)
  if m:
    dist = float(m.group(1))
    if dist > 0.55 or dist < 0.15:
      raise Exception("Invalid detector distance (%f m)" % dist)
    return
  raise Exception("No detector distance found in header!")

def test_cif(data):
  m = re.search('GON_PHI +rotation +goniometer GON_OMEGA 0.5774 -0.8165 0', data)
  if m:
    return
  raise Exception("Cif header incorrect, incomplete or missing!")

def test_gain(data):
  try:
    wl = test_wavelength(data, straight_return=True)
  except Exception:
    return

  m = re.search('# Tau = ([0-9.e+-]+) s', data)
  if not m:
    raise Exception("Tau not found in header")
  tau = float(m.group(1))

  m = re.search('# Gain_setting: ([a-z]+) gain \(vrf = ([0-9.+-]+)\)', data)
  if not m:
    raise Exception("Gain information not found in header")
  gain = m.group(1)
  vrf = float(m.group(2))

  m = re.search('# Threshold_setting: ([0-9]+) eV', data)
  if not m:
    raise Exception("Threshold information not found in header")
  actual_threshold = float(m.group(1))
  expected_threshold = (12398.42 / wl) / 2  # eV

  def test_values(exp_tau, exp_gain, exp_vrf):
    e = []
    if (gain != exp_gain):
      e.append('Gain (%s) deviates from expected setting (%s)' % (gain, exp_gain))
    if (abs((tau-exp_tau) / exp_tau) > 0.1):
      e.append('Tau (%e) deviates from expected tau (%e) by more than 10%%' % (tau, exp_tau))
    if (abs((vrf-exp_vrf) / exp_vrf) > 0.1):
      e.append('Vrf (%e) deviates from expected vrf (%e) by more than 10%%' % (vrf, exp_vrf))
    return e

  if (abs(expected_threshold - actual_threshold) / expected_threshold) > 0.1:
    raise Exception("Threshold (%f eV) deviates from expected threshold (%f eV) by more than 10%%" % \
      (actual_threshold, expected_threshold))
  if actual_threshold < 4000:
    raise Exception("Threshold (%f eV) outside detector specification (< 4 keV)" % actual_threshold)
  elif actual_threshold < 5000:
    return test_values(384e-9, 'high', -0.15)
  elif actual_threshold < 7000:
    return test_values(200e-9, 'mid', -0.2)
  elif actual_threshold <= 18000:
    return test_values(125e-9, 'low', -0.3)
  else:
    raise Exception("Threshold (%f eV) outside detector specification (> 18 keV)" % actual_threshold)

checks = { test_wavelength, test_beampos, test_distance, test_cif, test_gain }

for filename in sorted(filelist):
  with open(filename, 'rb') as infile:
    data = infile.read(4096 * 4)

  result = []
  for test in checks:
    try:
      messages = test(data)
      if messages is not None:
        result.extend(messages)
    except Exception, e:
      result.append(e.message)

  if result != []:
    print "\nFile %s failed the following checks:" % filename
    sys.stdout.write('\033[1;31m ')
    print "\n ".join(result)
    sys.stdout.write('\033[0m')
    print
    break

  print "Checking file %s" % filename


