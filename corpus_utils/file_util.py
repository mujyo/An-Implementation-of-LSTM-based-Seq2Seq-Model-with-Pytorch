def load_file(file_path): 
    """ A helper functions that loads the file into a tuple of strings
     :param file_path: path to the data file
     :return factors: (LHS) inputs to the model
             expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return list([list(factors), list(expansions)])


def read_txt(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def write_txt(file_path, content):
    with open(file_path, 'w') as f:
        for line in content:
            f.write("%s\n" % line)
