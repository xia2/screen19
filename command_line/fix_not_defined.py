from __future__ import division

def main():
  import sys

  value = float(sys.argv[1])
  undefined = '<not_defined>'
  print '"%s" -> %f in %d files' % (undefined, value, len(sys.argv[2:]))

  for filename in sys.argv[2:]:
    data = open(filename).read()
    assert(undefined in data)
    fixed = data.replace(undefined, sys.argv[1])
    open(filename, 'w').write(fixed)
    print '.',

main()
