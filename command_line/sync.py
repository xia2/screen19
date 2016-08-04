from libtbx.utils import Sorry
from optparse import OptionParser, SUPPRESS_HELP
import os
import shutil
import sys
import time

class FileCopier():
  def __init__(self, source, destination, maxwait=3*60*60, symlink=False):
    self._destination = destination
    self._source = source
    self._symlink = symlink
    self._timeout = time.time() + maxwait

  def __str__(self):
    return "File: %s" % self._source

  def __repr__(self):
    return self.__str__()

  def check(self):
    if os.path.exists(self._destination):
      print "Destination %s exists." % self._destination
      return False
    if os.path.exists(self._source):
      targetdir = os.path.dirname(self._destination)
      if not os.path.exists(targetdir):
        os.makedirs(targetdir)
      if self._symlink:
        print "Linking %s > %s" % (self._source, self._destination)
        os.symlink(self._source, self._destination)
      else:
        print "Copying %s > %s" % (self._source, self._destination)
        shutil.copyfile(self._source, self._destination)
      return False
    return time.time() < self._timeout

class FolderCopier():
  def __init__(self, source, destination, task_scheduler, maxwait=3*60*60):
    self._source = source
    self._destination = destination
    self._add_task = task_scheduler
    self._timeout = time.time() + maxwait
    self._seenfiles = set([])

  def __str__(self):
    return "Dir : %s" % self._source

  def __repr__(self):
    return self.__str__()

  def check(self):
    for f in os.listdir(self._source):
      if f not in self._seenfiles:
        self._seenfiles.add(f)
        self._add_task(FileCopier(
            os.path.join(self._source, f),
            os.path.join(self._destination, f),
            maxwait = 5*60))
#       print "Seen new file", os.path.join(self._source, f)
    return time.time() < self._timeout

class WaitForFolder():
  def __init__(self, folder, callback, args=[], maxwait=3*60*60):
    self._folder = folder
    self._callback = callback
    self._callback_args = args
    self._timeout = time.time() + maxwait

  def __str__(self):
    return "Wait: %s" % self._folder

  def __repr__(self):
    return self.__str__()

  def check(self):
    if os.path.exists(self._folder):
      print "Folder %s found" % self._folder
      self._callback(*self._callback_args)
      return False
    return time.time() < self._timeout

class FindProcessed():
  def __init__(self):
    self._basepath = '/dls/i19-1/data/2016'
    self._watch_tasks = []
    self._added_tasks = []

  def _find_new_processed_folders(self):
    def delayed_copy_structure_folder(source, destination):
      self.add_task(FolderCopier(source, destination, self.add_task, maxwait=1*60*60))

    while True:
      folder = self._processfile.readline()
      if folder:
        folder = folder.rstrip()
        source = folder
        destination = folder.replace('/processed/', '/processing/')
#       destination = destination.replace('/dls/i19-1/data/2016', '/dls/tmp/wra62962/copytest')

        # Symlink these files ASAP
        self.add_task(FileCopier(
            os.path.join(source, 'merging-statistics.txt'),
            os.path.join(destination, 'merging-statistics.txt'),
            symlink=True))

        self.add_task(FileCopier(
            os.path.join(source, 'xia2.txt'),
            os.path.join(destination, 'xia2.txt'),
            symlink=True))

        self.add_task(FileCopier(
            os.path.join(source, 'xia2.cif'),
            os.path.join(destination, 'xia2.cif'),
            symlink=True))

        # Wait for structure folder, if it appears monitor and copy everything in it.
        self.add_task(WaitForFolder(
            os.path.join(source, 'structure'),
            delayed_copy_structure_folder,
            [os.path.join(source, 'structure'), destination]
            ))
      else:
        break

  def run(self):
    parser = OptionParser(usage="usage: %prog [options] visitid")
#   parser.add_option("-v", action="store_true", dest="verbose", default=False,
#                     help="be moderately verbose")
    parser.add_option("-?", help=SUPPRESS_HELP, action="help")
    self._opts, args = parser.parse_args()
    if len(args) != 1:
      raise Sorry('You need to give exactly one visit number as parameter')
    self._visit = args[0]

    # Step 1: Find visit directory
    self._visitpath = os.path.join(self._basepath, self._visit)
    if not os.path.exists(self._visitpath):
      raise Sorry("Could not find visit %s at %s" % (self._visit, self._visitpath))
    print "Running for visit %s" % self._visit

    # Step 2: Find process.dirs file, or wait for up to 25 minutes for it to appear
    processdirsfile = os.path.join(self._visitpath, 'spool/process.dirs')
    maxwait = 25
    while not os.path.exists(processdirsfile):
      print "No data collections have been made in that visit folder yet."
      print "Waiting %d minutes for first data collection." % maxwait
      time.sleep(60)
      maxwait = maxwait - 1
      if maxwait <= 0:
        raise Sorry('No data collections observed for visit %s' % self._visit)
    self._processfile = open(os.path.join(self._visitpath, 'spool/process.dirs'), 'r')
    print "Data collection list found."

    # Step 3: Generate and process copy tasks
    self._task_loop()

  def _task_loop(self):
    # Limit script execution time to 24 hours
    maxtime = time.time() + 24 * 60 * 60
    while maxtime > time.time():
      remaining_tasks, self._added_tasks = [], []
      self._find_new_processed_folders()
      for f in self._watch_tasks:
        if f.check():
          remaining_tasks.append(f)
        time.sleep(0.01)
      self._watch_tasks = remaining_tasks + self._added_tasks
      print "\nWaiting to copy new results in %s (watching %d places)\n" % (self._visit, len(self._watch_tasks))
      time.sleep(30)

    print "Terminating script after one day."

  def add_task(self, task):
    self._added_tasks.append(task)

if __name__ == "__main__":
  FindProcessed().run()
