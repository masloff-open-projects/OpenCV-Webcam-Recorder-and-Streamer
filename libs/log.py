class log:

    use_file = False

    def info (self, status, text):

        print ('[{status:>2}]: {text}'.format(
            status=status,
            text=text
        ))
