from generator import Generator

def test_filename_generator():
  pattern = "./somefile_%05d.cbf"
  rng = (3, 18)

  g = Generator(pattern, rng[0], rng[1])
  gen = list(g)

  assert len(gen) == rng[1] - rng[0] + 1
  i = rng[0]
  for entry in gen:
    assert entry == pattern % i
    i += 1
