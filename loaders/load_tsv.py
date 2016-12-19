

def load_tsv(path):
    lines = []
    with open(path, 'r') as file:
        line = file.readline()
        while line != '':
            lines.append(line.split('\t'))
            line = file.readline()
    return lines
