import shutil
import dill
from zipfile import ZipFile
import os
import glob
import abc
import re

from dps.utils import zip_root


class ObjectFragment(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def combine(self, *others):
        raise Exception("NotImplemented")


class _ObjectWrapper(object, metaclass=abc.ABCMeta):
    def __init__(self, directory, kind, key):
        self.directory = directory
        self.kind = kind
        self.key = key

    @abc.abstractmethod
    def __str__(self):
        raise Exception("NotImplemented")

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def _indices(self):
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def _load_fragment(self, path):
        raise Exception("NotImplemented")

    def _filenames(self):
        """ Basenames of all files covered by this object. """
        for idx in sorted(self._indices()):
            if idx is None:
                basename = "{}.key".format(self.key)
            else:
                basename = "{}.{}.key".format(self.key, idx)
            yield basename

    def load(self):
        fragments = []
        for basename in self._filenames():
            path = os.path.join(self.directory, self.kind, basename)

            fragment = self._load_fragment(path)

            fragments.append(fragment)

        fragmented = len(fragments) > 1 or isinstance(fragments[0], ObjectFragment)

        if fragmented:
            non_fragment_types = [type(f) for f in fragments if not isinstance(f, ObjectFragment)]
            if non_fragment_types:
                raise ValueError(
                    "Object with kind {} and key {} is stored in fragmented format, but "
                    "some of the fragments are not instances of `ObjectFragment`. Types "
                    "are: {}.".format(self.kind, self.key, non_fragment_types))
            return fragments[0].combine(*fragments[1:])
        else:
            return fragments[0]

    def exists(self):
        return bool(list(self._indices()))


class _FileSystemObjectWrapper(_ObjectWrapper):
    """ An object stored on the file system using dill. """

    def __init__(self, directory, kind, key):
        self.directory = directory
        self.kind = kind
        self.key = key

    def __str__(self):
        return "_FileSystemObjectWrapper(directory={}, kind={}, key={})".format(self.directory, self.kind, self.key)

    def _indices(self):
        pattern = os.path.join(self.directory, self.kind, "{}*.key".format(self.key))
        for obj_path in glob.iglob(pattern):
            basename = os.path.basename(obj_path)
            parts = basename.split('.')
            if len(parts) == 2:
                idx = None
            else:
                idx = int(parts[1])
            yield idx

    def _load_fragment(self, path):
        with open(path, 'rb') as f:
            return dill.load(f)

    def _next_idx(self):
        return max(self._indices(), default=-1) + 1

    def add_fragment(self, fragment, recurse=False):
        idx = self._next_idx()
        path = os.path.join(self.directory, self.kind, "{}.{}.key".format(self.key, idx))

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            dill.dump(fragment, f, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)

    def delete(self):
        existed = self.exists()
        for filename in self._filenames():
            path = os.path.join(self.directory, self.kind, filename)
            os.remove(path)
        return existed


class _ZipObjectWrapper(_ObjectWrapper):
    """ An object stored in a zip file using dill. """

    def __init__(self, zip_file, directory, kind, key):
        self.zip_file = zip_file
        self.directory = directory
        self.kind = kind
        self.key = key

    def __str__(self):
        return "_ZipObjectWrapper(zip={}, directory={}, kind={}, key={})".format(
            self.zip, self.directory, self.kind, self.key)

    def _indices(self):
        pattern = os.path.join(self.directory, self.kind, "{}.+\.key".format(self.key))
        reg_exp = re.compile(pattern)
        for path in self.zip_file.namelist():
            if reg_exp.match(path):
                basename = os.path.basename(path)

                parts = basename.split('.')
                if len(parts) == 2:
                    idx = None
                else:
                    idx = int(parts[1])
                yield idx

    def _load_fragment(self, path):
        with self.zip_file.open(path, 'r') as f:
            return dill.load(f)


class ObjectStore(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _wrap_object(self, kind, key):
        """ Return an abstract object that wraps `kind` and `key`. """
        raise Exception("NotImplemented")

    @abc.abstractmethod
    def keys(self, kind):
        """ Return all keys for given kind. """
        raise Exception("NotImplemented")

    def get_unique_key(self, kind):
        """ Get a unique key for the given kind. """
        int_keys = [key for key in self.keys(kind) if isinstance(key, int)]
        return max(int_keys, default=-1) + 1

    def save_object(self, kind, key, obj, recurse=True):
        wrapped = self._wrap_object(kind, key)

        fragment = isinstance(obj, ObjectFragment)
        if not fragment and wrapped.exists():
            raise ValueError("Trying to save object {} with kind {} and key {}, "
                             "but an object {} already exists at that location."
                             "".format(obj, kind, key, self.load_object(kind, key)))

        wrapped.add_fragment(obj)

    def load_object(self, kind, key):
        obj = self._wrap_object(kind, key)
        return obj.load()

    def load_objects(self, kind):
        objects = {}
        for key in self.keys(kind):
            obj = self.load_object(kind, key)
            objects[key] = obj
        return objects

    def n_objects(self, kind=None):
        return len(self.keys(kind))

    def object_exists(self, kind, key):
        return key in self.keys(kind)


class FileSystemObjectStore(ObjectStore):
    """ Stores objects on the file system, using dill to serialize. """

    def __init__(self, directory):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        super(FileSystemObjectStore, self).__init__()

    def path_for_kind(self, kind):
        return os.path.join(self.directory, kind)

    def _wrap_object(self, kind, key):
        return _FileSystemObjectWrapper(self.directory, kind, key)

    def keys(self, kind):
        """ Return all keys for given kind. """
        keys = set()
        pattern = os.path.join(self.directory, kind, "*.key")
        for obj_path in glob.iglob(pattern):
            basename = os.path.basename(obj_path)
            key = basename.split('.')[0]
            try:
                key = int(key)
            except (ValueError, TypeError):
                pass
            keys.add(key)

        return sorted(keys)

    def delete_object(self, kind, key):
        wrapped = self._wrap_object(kind, key)
        return wrapped.delete()

    def zip(self, archive_path=None, delete=False):
        """ Zip the object store. """
        parent = os.path.dirname(self.directory)

        if not archive_path:
            archive_name = os.path.basename(self.directory)
            archive_path = os.path.join(parent, archive_name)

        archive_path = os.path.splitext(archive_path)[0]

        base_dir = os.path.basename(self.directory)

        archive_path = shutil.make_archive(
            archive_path, 'zip', root_dir=parent, base_dir=base_dir)

        print("Zipped {} as {}.".format(self.directory, archive_path))
        if delete:
            shutil.rmtree(self.directory)

        return archive_path


class ZipObjectStore(ObjectStore):
    """ A read-only object store based on a zip file. Avoids ever unzipping the entire file. """

    def __init__(self, zip_path):
        zip_path = os.path.splitext(zip_path)[0] + ".zip"
        self.zip_file = ZipFile(zip_path, 'r')
        self.directory = zip_root(zip_path)

    def path_for_kind(self, kind):
        return os.path.join(self.directory, kind)

    def _wrap_object(self, kind, key):
        return _ZipObjectWrapper(self.zip_file, self.directory, kind, key)

    def keys(self, kind):
        keys = set()
        pattern = os.path.join(self.directory, kind, ".+\.key")
        reg_exp = re.compile(pattern)
        for path in self.zip_file.namelist():
            if reg_exp.match(path):
                basename = os.path.basename(path)
                key = basename.split('.')[0]
                try:
                    key = int(key)
                except (ValueError, TypeError):
                    pass
                keys.add(key)

        return sorted(keys)
