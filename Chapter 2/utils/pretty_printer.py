from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def inspect_variables(variables):
    for var in variables:
            print('name = {} {}shape = {}'.format(var.name, " "*(55-len(var.name)), var.get_shape()))
    print()

def inspect_layers(endpoints):
    for k, v in endpoints.iteritems():
        print('name = {} {}shape = {}'.format(v.name, " "*(55-len(v.name)), v.get_shape()))
    print()
