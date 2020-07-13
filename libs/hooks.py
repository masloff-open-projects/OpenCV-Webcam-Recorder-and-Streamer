class hooks:

    def __init__(self):
        self.hooks = {}

    def set (self, name, function):
        if not name in self.hooks:
            self.hooks[name] = []
            self.hooks[name].append(function)
        else:
            self.hooks[name].append(function)

    def get (self, name):
        if name in self.hooks:
            return self.hooks[name]

    def call (self, name, argv):

        if (not self.get(name) == None):

            for _ in self.get(name):
                try:
                    if callable(_):
                        _(argv)
                except: pass
