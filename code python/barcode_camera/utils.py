#!/usr/bin/env python

import os.path


class Atter(object):
    pass


def upper_dir(path, level):
    for _ in range(level):
        path = os.path.dirname(path)
    return path


def module_to_class_name(name):
    return "".join(i.title() for i in name.split("_"))


def class_to_module_name(name):
    return_value = name[0].lower()
    for i in name[1:]:
        return_value += "_%s" % (i.lower(),) if "A" <= i <= "Z" else i
    return return_value


def to_list(*args):
    if len(args) != 1:
        return args
    args = args[0]
    if isinstance(args, (list, tuple)):
        return args
    return [args]

def hasattrs(obj, *attrs):
    for attr in to_list(*attrs):
        if not hasattr(obj, attr):
            return False
    return True


def delattrs(obj, *attrs):
    for attr in to_list(*attrs):
        if hasattr(obj, attr):
            delattr(obj, attr)

def cpyattrs(src, dst, *attrs):
    for attr in to_list(*attrs):
        setattr(dst, attr, getattr(src, attr))


def get_exposed_attrs(obj):
    return [i for i in dir(obj) if not i.endswith('_')]