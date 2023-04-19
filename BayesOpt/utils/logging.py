import os, pickle

def check_log(dirpath, filename):
    """
    Check log includes x and y output.

    :param dirpath: pkl file path of the logfile.
    :param filename: pkl name.
    :param eta: eta parameter.
    """
    file = os.path.join(dirpath, filename)
    try:
        return os.path.isfile(file)
    except FileNotFoundError:
        return False

def create_log(dirpath:str, filename:str, outputs: list):
    '''
    Creates a new log file.
    
    :param dirpath: Path to the directory to be saved.
    :param filename: Name of logfile.
    :return result: A dictionary of results.
    '''
    os.makedirs(dirpath, exist_ok=True)
    file = os.path.join(dirpath, filename)
    with open(file, "wb") as outfile:
        pickle.dump(outputs, outfile)
    print(f"Log created for {filename}")
